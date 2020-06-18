from config import configurator

from argument.tfrs import TFRsPerformingArguments,TFRsDataArguments,TFRsModelArguments
from data.proxy.tfrs import TFRsDataProxy
from framework.transformers.run_glue import TFRsFrameworkProxy

configurator._return_remaining_strings = True
configurator.register_arguments(model_args = TFRsModelArguments, data_args=TFRsDataArguments, performing_args=TFRsPerformingArguments)
configurator.data_proxy_type = TFRsDataProxy
configurator.framework_proxy_type = TFRsFrameworkProxy

# from argument.mtl_pi import MTLPIPerformingArguments, MTLPIModelArguments, MTLPIDataArguments
# from data.proxy.mtl_pi import MTLPIDataProxy
# from framework.mtl.mtl_pi import MTLPIFrameworkProxy
#
# configurator.register_arguments(model_args=MTLPIModelArguments,
#                                 data_args=MTLPIDataArguments,
#                                 performing_args=MTLPIPerformingArguments)
#
# configurator.data_proxy_type = MTLPIDataProxy
# configurator.framework_proxy_type = MTLPIFrameworkProxy


import controller


c = controller.Controller()
c.run()