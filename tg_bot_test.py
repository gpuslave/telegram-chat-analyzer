import json

import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler

# reading json file that contains secret token to a bot
with open(r'variables.json', encoding='utf-8') as json_token:
    json_str = json_token.read()
    var_dic = json.loads(json_str)
    token = var_dic["token"]

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pass