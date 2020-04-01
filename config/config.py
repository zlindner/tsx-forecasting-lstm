# load .env
from dotenv import load_dotenv
load_dotenv()

# default config
default_config = {

}

# build config
import os
from config.dev import config as dev_config
from config.prod import config as prod_config

env = os.getenv('PYTHON_ENV') or 'dev'
env_config = None

if env == 'dev':
    env_config = dev_config
elif env == 'prod':
    env_config = prod_config

def chain_configs(*config_items):
    for it in config_items:
        for element in it:
            value = element[1]
            
            if value is not None:
                yield element

config = dict(chain_configs(*default_config.items(), env_config.items()))