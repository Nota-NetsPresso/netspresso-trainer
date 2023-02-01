import os
import logging
import requests

DEPLOYMENT_MODE = os.environ.get("DEPLOYMENT_MODE")
if DEPLOYMENT_MODE and DEPLOYMENT_MODE == "PROD":
    MODELSEARCH_API_SERVER_BASE_URI = "https://searcher.netspresso.ai/api/v1"  # PROD mode
else:
    MODELSEARCH_API_SERVER_BASE_URI = "http://1.235.98.12:40001/api/v1"  # DEV mode

STATUS_CODE_400 = 400

_logger = logging.getLogger(__name__)

class ModelSearchServerHandler:
    def __init__(self, project_id: str, token: str):
        self.project_id = project_id
        self.token = token
        self.only_once = True
        self.is_sent = False
        self.dest = f"{MODELSEARCH_API_SERVER_BASE_URI}/project/{self.project_id}"

    def report_elapsed_time_for_epoch(self, elapsed_time_one_epoch: int):
        if not (self.only_once and self.is_sent):
            try:
                headers = {
                    'Authorization': f"Bearer {self.token}",
                }
                response = requests.patch(self.dest, json={"elapsed_time": elapsed_time_one_epoch}, headers=headers)

                _logger.info(f"Status Code: {response.status_code}")
                _logger.debug(f"Content: {response.content}")

                if response.status_code >= STATUS_CODE_400:
                    raise Exception("To estimate elapsed time of one epoch failed.")
            except Exception as e:
                _logger.error(e)
            finally:
                self.is_sent = True
