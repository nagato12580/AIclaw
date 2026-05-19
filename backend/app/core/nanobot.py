import asyncio
import sys
import os
import shutil
from pathlib import Path
from typing import List, Callable, Awaitable, Any, Dict

# Add project root to sys.path to allow importing nanobot
# Assuming backend/app/core/nanobot.py -> backend/app/core -> backend/app -> backend -> root
# This path calculation seems correct for backend/app/core/nanobot.py relative to backend/
# BUT nanobot package is in ../nanobot relative to backend/
# So we need to go up one more level to reach the parent of backend/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "nanobot") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "nanobot"))

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.loader import load_config
from nanobot.cron.service import CronService
from nanobot.providers.openai_codex_provider import OpenAICodexProvider
from nanobot.providers.azure_openai_provider import AzureOpenAIProvider
from nanobot.providers.base import GenerationSettings
from nanobot.providers.registry import find_by_name
from nanobot.session.manager import SessionManager
from nanobot.config.schema import Config

# Import skills loader
# We use a lazy import inside the method to avoid potential circular dependencies if any arise,
# or just import here if we are confident.
# Given the structure, importing here should be fine as long as skills.py doesn't import nanobot.py.
from app.api.skills import load_skills
from app.core.patched_openai_compat_provider import PatchedOpenAICompatProvider
from app.services.llm_cache import get_llm_configs, get_active_llm_config
from app.services.web_search_config_store import get_web_search_config

from app.core.data_root import get_workspace_root
from app.trace import build_error_attributes, build_usage_attributes, trace_service

class NanobotIntegration:
    """Nanobot 集成服务类，负责管理 Agent、消息总线、定时任务等核心组件的生命周期。"""

    def __init__(self):
        """初始化 NanobotIntegration 实例，创建所有核心组件的空引用。"""
        # 主代理循环实例
        self.agent: AgentLoop | None = None
        # 消息总线，用于组件间通信
        self.bus: MessageBus | None = None
        # 定时任务服务
        self.cron: CronService | None = None
        # 配置对象
        self.config: Config | None = None
        # 服务启动状态标志
        self._started = False
        # 模型代理缓存：key 为 (model_id, project_id)，value 为对应的 AgentLoop 实例
        self._model_agent_cache: Dict[tuple[str | None, int | None], AgentLoop] = {}
        # 模型代理缓存的异步锁，确保并发安全
        self._model_agent_lock = asyncio.Lock()
        # 按会话记录的最后使用情况（token 使用量等）
        self._last_usage_by_session: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _normalize_config_value(value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @staticmethod
    def _normalize_model_id(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return str(value)

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if isinstance(response, OutboundMessage):
            return response.content or ""
        if isinstance(response, dict):
            content = response.get("content")
            if isinstance(content, str):
                return content
            return str(content or "")
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content
        return str(response)

    @staticmethod
    def _normalize_usage(usage: Any) -> Dict[str, int] | None:
        if not isinstance(usage, dict):
            return None
        normalized: Dict[str, int] = {}
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", 0) or 0)
        
        # If total_tokens is missing or zero, calculate it
        if total == 0:
            total = prompt + completion
            
        normalized["prompt_tokens"] = prompt
        normalized["completion_tokens"] = completion
        normalized["total_tokens"] = total
        return normalized if (prompt > 0 or completion > 0) else None

    def get_last_usage(self, session_id: str) -> Dict[str, int] | None:
        usage = self._last_usage_by_session.get(session_id)
        return dict(usage) if usage else None

    def _get_web_search_config(self) -> Any:
        from nanobot.config.schema import WebSearchConfig
        ws_dict = get_web_search_config()
        return WebSearchConfig(
            provider=ws_dict.get("provider", "duckduckgo"),
            api_key=ws_dict.get("api_key", ""),
            base_url=ws_dict.get("base_url", ""),
            max_results=ws_dict.get("max_results", 5)
        )

    def _need_custom_agent_for_target(self, target_config: Dict[str, Any]) -> bool:
        if not self.agent:
            return False

        provider = self.agent.provider
        target_model = self._normalize_config_value(target_config.get("model"))
        current_model = self._normalize_config_value(
            getattr(self.agent, "model", None) or getattr(provider, "default_model", None)
        )
        if target_model != current_model:
            return True

        target_provider = self._normalize_config_value(target_config.get("provider"))
        current_provider = self._normalize_config_value(getattr(provider, "_provider_name_override", None))
        if not current_provider:
            current_provider = self._normalize_config_value(getattr(getattr(provider, "_spec", None), "name", None))
        if not current_provider and current_model and self.config:
            current_provider = self._normalize_config_value(self.config.get_provider_name(current_model))
        if target_provider != current_provider:
            return True

        target_api_base = self._normalize_config_value(target_config.get("api_base"))
        current_api_base = self._normalize_config_value(getattr(provider, "api_base", None))
        if target_api_base != current_api_base:
            return True

        target_api_key = self._normalize_config_value(target_config.get("api_key"))
        current_api_key = self._normalize_config_value(getattr(provider, "api_key", None))
        if target_api_key != current_api_key:
            return True

        target_headers = target_config.get("extra_headers") or {}
        current_headers = getattr(provider, "extra_headers", None) or {}
        return target_headers != current_headers

    def initialize(self):
        workspace_path = get_workspace_root()
        workspace_path.mkdir(parents=True, exist_ok=True)
        self._sync_builtin_skills_to_workspace(workspace_path)
        
        # Override config workspace path via environment variable (since config is loaded from env)
        os.environ["NANOBOT_AGENTS__DEFAULTS__WORKSPACE"] = str(workspace_path)
        
        self.config = load_config()
        # No need to set self.config.workspace_path as it's a property that reads from agents.defaults.workspace
        
        self.bus = MessageBus()
        active_config = get_active_llm_config()
        initial_model = self.config.agents.defaults.model
        if active_config and active_config.get("model"):
            provider = self._make_provider_from_target(active_config)
            initial_model = self._normalize_config_value(active_config.get("model")) or initial_model
        else:
            provider = self._make_provider(self.config)
        
        cron_store_path = workspace_path / "cron"
        cron_store_path.mkdir(parents=True, exist_ok=True)
        cron_store_file = cron_store_path / "jobs.json"
        
        self.cron = CronService(cron_store_file)
        
        session_manager = SessionManager(self.config.workspace_path)

        self.agent = AgentLoop(
            bus=self.bus,
            provider=provider,
            workspace=self.config.workspace_path,
            model=initial_model,
            max_iterations=self.config.agents.defaults.max_tool_iterations,
            context_window_tokens=self.config.agents.defaults.context_window_tokens,
            web_search_config=self._get_web_search_config(),
            web_proxy=self.config.tools.web.proxy or None,
            exec_config=self.config.tools.exec,
            cron_service=self.cron,
            restrict_to_workspace=self.config.tools.restrict_to_workspace,
            session_manager=session_manager,
            mcp_servers=self.config.tools.mcp_servers,
            channels_config=self.config.channels,
            timezone=self.config.agents.defaults.timezone,
        )

        self._register_custom_tools(self.agent)

    def _sync_builtin_skills_to_workspace(self, workspace_path: Path) -> None:
        """
        将内置技能同步到工作区。
        
        该方法负责将项目内置的技能（如 nl2sql、visualization）复制到工作区的 skills 目录下，
        以便 Agent 能够发现和使用这些技能。
        
        Args:
            workspace_path: 工作区根目录路径
        """
        # 内置技能的源目录（相对于当前文件的父目录下的 skills_builtin）
        builtin_root = Path(__file__).resolve().parents[1] / "skills_builtin"
        # 工作区技能目录
        workspace_skills_root = workspace_path / "skills"
        # 确保目标目录存在
        workspace_skills_root.mkdir(parents=True, exist_ok=True)

        # 遍历需要同步的内置技能
        for skill_name in ("nl2sql", "visualization"):
            source_dir = builtin_root / skill_name
            source_skill_file = source_dir / "SKILL.md"
            # 如果源技能文件不存在，跳过
            if not source_skill_file.exists():
                continue
            # 创建目标技能目录
            target_dir = workspace_skills_root / skill_name
            target_dir.mkdir(parents=True, exist_ok=True)
            # 复制技能文件（保留元数据）
            shutil.copy2(source_skill_file, target_dir / "SKILL.md")

    def _register_custom_tools(self, agent: AgentLoop, project_id: int | None = None):
        from app.tools.nl2sql import NL2SQLTool
        from app.tools.visualization import VisualizationTool
        from app.tools.get_schema import GetDatabaseSchemaTool
        from app.tools.knowledge_base import KnowledgeBaseRetrieveTool
        from app.tools.subagent import ListSubagentsTool, InvokeSubagentTool
        agent.tools.register(NL2SQLTool())
        agent.tools.register(VisualizationTool())
        agent.tools.register(GetDatabaseSchemaTool())
        agent.tools.register(KnowledgeBaseRetrieveTool())
        agent.tools.register(ListSubagentsTool(project_id=project_id))
        agent.tools.register(InvokeSubagentTool(project_id=project_id))

    def _build_provider(
        self,
        model: str,
        provider_name: str | None,
        api_key: str | None,
        api_base: str | None,
        extra_headers: dict[str, Any] | None = None,
    ):
        spec = find_by_name(provider_name) if provider_name else None
        backend = spec.backend if spec else "openai_compat"

        if backend == "openai_codex" or model.startswith("openai-codex/"):
            return OpenAICodexProvider(default_model=model)

        if backend == "azure_openai":
            if not api_key or not api_base:
                raise ValueError("Azure OpenAI requires api_key and api_base.")
            return AzureOpenAIProvider(
                api_key=api_key,
                api_base=api_base,
                default_model=model,
            )

        if backend == "anthropic":
            from nanobot.providers.anthropic_provider import AnthropicProvider
            return AnthropicProvider(
                api_key=api_key,
                api_base=api_base,
                default_model=model,
                extra_headers=extra_headers,
            )

        return PatchedOpenAICompatProvider(
            api_key=api_key,
            api_base=api_base,
            default_model=model,
            extra_headers=extra_headers,
            spec=spec,
        )

    def _make_provider(self, config: Config):
        model = config.agents.defaults.model
        provider_name = config.get_provider_name(model)
        p = config.get_provider(model)
        provider = self._build_provider(
            model=model,
            provider_name=provider_name,
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            extra_headers=p.extra_headers if p else None,
        )
        provider.generation = GenerationSettings(
            temperature=config.agents.defaults.temperature,
            max_tokens=config.agents.defaults.max_tokens,
            reasoning_effort=config.agents.defaults.reasoning_effort,
        )
        return provider

    def _make_provider_from_target(self, target_config: Dict[str, Any]):
        model = self._normalize_config_value(target_config.get("model")) or self.config.agents.defaults.model
        provider_name = self._normalize_config_value(target_config.get("provider"))
        if not provider_name and model and self.config:
            provider_name = self._normalize_config_value(self.config.get_provider_name(model))
        provider = self._build_provider(
            model=model,
            provider_name=provider_name,
            api_key=self._normalize_config_value(target_config.get("api_key")),
            api_base=self._normalize_config_value(target_config.get("api_base")),
            extra_headers=target_config.get("extra_headers"),
        )
        provider.generation = GenerationSettings(
            temperature=self.config.agents.defaults.temperature,
            max_tokens=self.config.agents.defaults.max_tokens,
            reasoning_effort=self.config.agents.defaults.reasoning_effort,
        )
        return provider

    async def start(self):
        if self._started:
            return
        if not self.agent:
            self.initialize()
        asyncio.create_task(self.agent.run())
        asyncio.create_task(self.cron.start())
        self._started = True

    async def stop(self):
        if self.agent:
            self.agent.stop()
            await self.agent.close_mcp()
        for agent in self._model_agent_cache.values():
            agent.stop()
            await agent.close_mcp()
        self._model_agent_cache.clear()
        if self.cron:
            self.cron.stop()
        self._started = False

    def _build_agent_for_provider(self, provider: Any, mcp_servers: dict | None = None) -> AgentLoop:
        return AgentLoop(
            bus=self.bus,
            provider=provider,
            workspace=self.config.workspace_path,
            model=provider.default_model,
            max_iterations=self.config.agents.defaults.max_tool_iterations,
            context_window_tokens=self.config.agents.defaults.context_window_tokens,
            web_search_config=self._get_web_search_config(),
            web_proxy=self.config.tools.web.proxy or None,
            exec_config=self.config.tools.exec,
            cron_service=self.cron,
            restrict_to_workspace=self.config.tools.restrict_to_workspace,
            session_manager=self.agent.sessions if self.agent else None,
            mcp_servers=mcp_servers if mcp_servers is not None else self.config.tools.mcp_servers,
            channels_config=self.config.channels,
            timezone=self.config.agents.defaults.timezone,
        )

    async def _get_or_create_model_agent(self, model_id: str | None, target_config: Dict[str, Any] | None, project_id: int | None = None) -> AgentLoop:
        normalized_model_id = self._normalize_model_id(model_id)
        cache_key = (normalized_model_id, project_id)
        async with self._model_agent_lock:
            cached = self._model_agent_cache.get(cache_key)
            if cached:
                return cached
            
            if target_config:
                provider = self._make_provider_from_target(target_config)
            else:
                provider = self._make_provider(self.config)

            mcp_servers_dict = dict(self.config.tools.mcp_servers) if self.config.tools.mcp_servers else {}
            if project_id is not None:
                from app.api.mcp import list_mcp_servers
                from nanobot.config.schema import MCPServerConfig
                servers = await list_mcp_servers(project_id=project_id)
                for s in servers:
                    cfg = MCPServerConfig(
                        type=s.get("type"),
                        command=s.get("command") or "",
                        args=s.get("args") or [],
                        env=s.get("env") or {},
                        url=s.get("url") or "",
                        headers=s.get("headers") or {}
                    )
                    mcp_servers_dict[s["name"]] = cfg

            agent = self._build_agent_for_provider(provider, mcp_servers=mcp_servers_dict)
            self._register_custom_tools(agent, project_id=project_id)
            self._model_agent_cache[cache_key] = agent
            return agent

    async def process_message(
        self,
        message: str,
        session_id: str = "api:default",
        skill_ids: List[str] | None = None,
        model_id: str | None = None,
        project_id: int | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
    ):
        """
        处理用户消息，返回 AI 响应。
        
        核心流程：
        1. 初始化并启动服务（如未启动）
        2. 从会话别名解析项目ID（如未提供）
        3. 根据模型ID和项目ID决定是否需要自定义Agent
        4. 获取或创建对应的Agent实例
        5. 标准化会话消息
        6. 调用Agent处理消息
        7. 提取响应文本和Token使用量
        
        Args:
            message: 用户输入的消息文本
            session_id: 会话ID，默认值为 "api:default"
            skill_ids: 技能ID列表（当前未使用）
            model_id: 模型ID，用于选择特定的LLM配置
            project_id: 项目ID，用于隔离不同项目的MCP服务器
            on_progress: 进度回调函数，接收进度消息
            on_stream: 流式输出回调函数，接收流式响应片段
            
        Returns:
            str: AI 响应文本
            
        Raises:
            Exception: 处理过程中发生的任何异常
        """
        # 设置追踪属性
        span_attributes = {
            "session_id": session_id,
            "project_id": project_id,
            "model_id": model_id,
            "component": "nanobot.process_message",
        }
        
        # 开始追踪根Span
        with trace_service.start_span(
            "nanobot.process_message",
            attributes=span_attributes,
            input_payload={"message": message},
        ) as root_span:
            try:
                # 初始化Agent（如未初始化）
                if not self.agent:
                    self.initialize()
                
                # 启动服务（如未启动）
                if not self._started:
                    await self.start()

                # 从会话别名解析项目ID（如果未直接提供）
                if project_id is None:
                    from app.core.session_alias_store import session_alias_store
                    alias_meta = session_alias_store.get_alias_meta(session_id)
                    if alias_meta and alias_meta.get("project_id") is not None:
                        project_id = alias_meta.get("project_id")
                        root_span.set_attributes({"project_id": project_id})

                # 默认使用主Agent
                agent_to_use = self.agent
                need_custom_agent = False
                target_config = None

                # 根据model_id查找对应的LLM配置
                selected_model_id = self._normalize_model_id(model_id)
                if selected_model_id:
                    llm_configs = get_llm_configs()
                    target_config = next(
                        (item for item in llm_configs if self._normalize_model_id(item.get("id")) == selected_model_id),
                        None,
                    )

                # 如果未找到目标配置，使用当前激活的配置
                if target_config is None:
                    active_config = get_active_llm_config()
                    if active_config and active_config.get("id"):
                        selected_model_id = self._normalize_model_id(active_config.get("id"))
                        target_config = active_config

                # 判断是否需要创建自定义Agent
                if target_config and self._need_custom_agent_for_target(target_config):
                    need_custom_agent = True
                # 有项目ID时也需要自定义Agent（隔离MCP服务器）
                if project_id is not None:
                    need_custom_agent = True

                # 解析并获取Agent实例
                with trace_service.start_span(
                    "nanobot.resolve_agent",
                    attributes={
                        "session_id": session_id,
                        "project_id": project_id,
                        "selected_model_id": selected_model_id,
                        "custom_agent": need_custom_agent,
                    },
                ):
                    if need_custom_agent:
                        agent_to_use = await self._get_or_create_model_agent(selected_model_id, target_config, project_id)

                # 获取或创建会话，并标准化消息格式
                session = agent_to_use.sessions.get_or_create(session_id)
                normalized_messages = self._normalize_session_messages(session.messages)
                if len(normalized_messages) != len(session.messages):
                    session.messages = normalized_messages
                    agent_to_use.sessions.save(session)

                # 调用Agent处理消息
                with trace_service.start_span(
                    "nanobot.process_direct",
                    attributes={
                        "session_id": session_id,
                        "model": getattr(agent_to_use, "model", None),
                    },
                ) as direct_span:
                    response = await agent_to_use.process_direct(
                        message,
                        session_key=session_id,
                        channel="api",
                        chat_id=session_id,
                        on_progress=on_progress,
                        on_stream=on_stream,
                    )
                    
                    # 提取并记录Token使用量
                    usage = self._normalize_usage(getattr(agent_to_use, "_last_usage", None))
                    if usage:
                        self._last_usage_by_session[session_id] = usage
                        direct_span.set_attributes(build_usage_attributes(usage))
                        root_span.set_attributes(build_usage_attributes(usage))
                    
                    # 提取响应文本
                    text = self._extract_response_text(response)
                    direct_span.update(output={"content": text})
                    root_span.update(output={"content": text})
                    return text
            except Exception as exc:
                # 记录错误到追踪系统
                root_span.set_attributes(build_error_attributes(exc, stage="nanobot_process_message"))
                root_span.record_error(exc, stage="nanobot_process_message")
                raise

    def _normalize_session_messages(self, messages: List[Any]) -> List[dict[str, Any]]:
        """
        标准化会话消息列表，将嵌套的列表结构扁平化。
        
        该方法使用广度优先遍历，处理可能嵌套的消息列表，
        只保留字典类型的消息对象，忽略其他类型。
        
        Args:
            messages: 原始消息列表，可能包含嵌套的列表结构
            
        Returns:
            List[dict[str, Any]]: 扁平化后的消息列表，仅包含字典类型的消息
        """
        normalized: List[dict[str, Any]] = []
        # 使用栈进行广度优先遍历
        stack: List[Any] = list(messages)
        
        while stack:
            # 弹出栈首元素
            current = stack.pop(0)
            
            # 如果是字典类型，直接添加到结果
            if isinstance(current, dict):
                normalized.append(current)
                continue
            
            # 如果是列表类型，将其元素插入到栈首（广度优先）
            if isinstance(current, list):
                stack = list(current) + stack
        
        return normalized

nanobot_service = NanobotIntegration()