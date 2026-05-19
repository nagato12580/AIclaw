import sys
print('sys.executable=', sys.executable)
try:
    import prometheus_client
    print('prometheus_client at', prometheus_client.__file__)
except Exception as e:
    print('import error:', e)
