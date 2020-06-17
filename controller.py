from config import configurator
from data.proxy import create_data_proxy
from framework import create_framework_proxy
import logging
from utils.general_tool import setup_seed
logger = logging.getLogger(__name__)


class Controller:
    def __init__(self):
        self.arguments_box = configurator.get_arguments_box()
        self._set_base_logger()
        logger.warning(
            "Device: %s, n_gpu: %s, 16-bits training: %s",
            self.arguments_box.performing_args.device,
            self.arguments_box.performing_args.n_gpu,
            self.arguments_box.performing_args.fp16,
        )
        logger.info("Arguments: %s", self.arguments_box)
        setup_seed(self.arguments_box.model_args.seed)

        self.data_proxy = create_data_proxy(configurator.data_proxy_type, self.arguments_box.data_args)


        self.framework_proxy = create_framework_proxy(configurator.framework_proxy_type, self.arguments_box.model_args,
                                                      self.arguments_box.performing_args, self.data_proxy)

        pass

    def _set_base_logger(self):
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

    def run(self):
        self.framework_proxy.perform(self.data_proxy)

