import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Trend } from "k6/metrics";

export const options = {
  scenarios: {
    default: {
      executor: "constant-vus",
      vus: Number(__ENV.VUS || 20),
      duration: __ENV.DURATION || "60s",
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.05"],
    http_req_duration: ["p(95)<10000"],
  },
};

const requestCounter = new Counter("dataclaw_requests_total");
const chatLatency = new Trend("dataclaw_chat_latency_ms");

const baseUrl = (__ENV.BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const endpoint = __ENV.ENDPOINT || "/nanobot/chat";
const url = `${baseUrl}${endpoint}`;
const token = __ENV.TOKEN || "";
const sessionPrefix = __ENV.SESSION_PREFIX || "k6";
const source = __ENV.SOURCE || "postgres";
const modelId = __ENV.MODEL_ID || "";
const projectId = __ENV.PROJECT_ID ? Number(__ENV.PROJECT_ID) : undefined;
const preferSqlChart = (__ENV.PREFER_SQL_CHART || "false").toLowerCase() === "true";
const fileUrl = __ENV.FILE_URL || "";
const routeMode = __ENV.ROUTE_MODE || "auto";
const message = __ENV.MESSAGE || "请统计最近 7 天的订单数，并给出结果。";
const streamExpected = endpoint.includes("/stream");

function buildPayload(iteration) {
  const payload = {
    message,
    session_id: `${sessionPrefix}-${__VU}-${__ITER}-${iteration}`,
    source,
    prefer_sql_chart: preferSqlChart,
    route_mode: routeMode,
  };

  if (projectId !== undefined && !Number.isNaN(projectId)) {
    payload.project_id = projectId;
  }
  if (modelId) {
    payload.model_id = modelId;
  }
  if (fileUrl) {
    payload.file_url = fileUrl;
  }

  return JSON.stringify(payload);
}

export default function () {
  const params = {
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    timeout: __ENV.TIMEOUT || "120s",
  };

  const started = Date.now();
  const response = http.post(url, buildPayload(Date.now()), params);
  const latency = Date.now() - started;
  chatLatency.add(latency);
  requestCounter.add(1);

  check(response, {
    "status is 2xx": (res) => res.status >= 200 && res.status < 300,
    "has response body": (res) => typeof res.body === "string" && res.body.length > 0,
  });

  if (!streamExpected) {
    check(response, {
      "has response field": (res) => {
        try {
          const data = JSON.parse(res.body);
          return typeof data.response === "string" || typeof data.detail === "string" || typeof data.error === "string";
        } catch (_e) {
          return false;
        }
      },
    });
  }

  sleep(Number(__ENV.SLEEP || 1));
}

export function handleSummary(data) {
  return {
    stdout: JSON.stringify(
      {
        metrics: {
          requests: data.metrics.iterations ? data.metrics.iterations.count : 0,
          http_req_failed: data.metrics.http_req_failed ? data.metrics.http_req_failed.passes / (data.metrics.http_reqs ? data.metrics.http_reqs.count : 1) : undefined,
          http_req_duration_p95: data.metrics.http_req_duration ? data.metrics.http_req_duration["p(95)"] : undefined,
          http_req_duration_avg: data.metrics.http_req_duration ? data.metrics.http_req_duration.avg : undefined,
          dataclaw_chat_latency_avg: data.metrics.dataclaw_chat_latency_ms ? data.metrics.dataclaw_chat_latency_ms.avg : undefined,
        },
      },
      null,
      2
    ),
  };
}
