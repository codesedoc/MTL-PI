from config import configurator
from argument.tfrs import TFRsPerformingArguments,TFRsDataArguments,TFRsModelArguments
from data.proxy.tfrs import TFRsDataProxy
from framework.transformers.run_glue import TFRsFrameworkProxy

configurator.register_arguments(model_args = TFRsModelArguments, data_args=TFRsDataArguments, performing_args=TFRsPerformingArguments)
configurator.data_proxy_type = TFRsDataProxy
configurator.framework_proxy_type = TFRsFrameworkProxy


import controller


c = controller.Controller()
c.run()