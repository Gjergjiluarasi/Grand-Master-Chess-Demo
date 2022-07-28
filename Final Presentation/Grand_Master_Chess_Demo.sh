#! /bin/sh
cd ..
echo "--------------------------- Welcome! ---------------------------
"
echo "    Choose game options:
    -g game_path (to change default game video path)
    -v 1 (to show results during the game)
    -s 1 (to save images during the game)
    "
read -p "
Do you want to play default game?    " choice 
if [ $choice = "y" ]
then
  echo "
  Starting default game...
  " 
  python main_interactive.py
elif [ $choice = "n" ]
then 
  read -p "
  Type your custom game options
  " custom_game
  $custom_game 
else
  echo "Wrong input"
fi
