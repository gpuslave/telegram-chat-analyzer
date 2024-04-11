# Telegram Chat Analyzer by gpuslave
### Features
* (IN FUTURE) A Telegram bot.
* Takes .json file that User needs to export via Telegram app from selected chat.
* Outputs a table of the most used words in this chat by each User, and also a comparison table.
  * (IN FUTURE) Also can output a .png plot of preceding stats via Seaborn plots.
  * (IN FUTURE) Also can output stats of most used swearwords individually.
* (IN FUTURE) Uses ML to distinguish laughter messages.

### How to run
1. You need to export your telegram chat history using build-in telegram functionality (export machine-readable json file)
2. Run the script (**alg.py**) with one argument, that is the path to this json file (usually something like C:\Users\User\Downloads\Telegram Desktop\ChatExport_YYYY-MM-DD\result.json)
