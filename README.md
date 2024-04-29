## Telegram Chat Analyzer

*--A smol pet proj that I made with passion--*
  
You like programming and chat with your friends? Then this script is all you need!

## Features

* Outputs a table of the most used words in this chat by each User, and also their comparison table.
* (under development) Distinguishes all laughter words (e.g. "haha") in one category.

## Installation (Win)

*Requires git and python 3.x (preferably latest) installed on your machine*
```cmd
git clone https://github.com/gpuslave/telegram-chat-analyzer.git
cd telegram-chat-analyzer
python -m venv .\.venv
.\\.venv\Scripts\Activate
pip install pandas matplotlib seaborn
```

## Documentation (Win)

1. You need to export chosen Telegram personal chat (dialogue) history using build-in telegram functionality (check all the boxes *off* -> change format to *machine-readable json* -> you can also increase the size limit if you have enormous amount of messages)
2. Run the script by `python alg.py`, while being in a virtual environment (run `.\\.venv\Scripts\Activate`)
3. Enter full path of the exported json file (by default `C:\Users\User\Downloads\Telegram Desktop\ChatExport_YYYY-MM-DD\result.json` on Windows)
