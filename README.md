# Grand-Master-Chess-Demo
A computer vision application for chess game activity recognition. Using YOLOv3 and Mediapipe the Grand Master Chess Demo was designed and implemented to capture and track in real time a chess game using camera input.

## Running the game
Please run the following line in the command line of a linux environment:

### Option 1) basic game
> python main.py 
### Option 2) interactive game - pause on Spacebar press
> python main_interactive.py 
### Option 3) game with arguments
-g "CUSTOM_GAME_PATH" -s 1 -v 1 
-g set custom game path
-s saves images during game
-v view results during game
> python main.py -g chess_v4/obj_test_data -s 1 -v 1

> python main.py -g chess_v4/obj_test_data -s 1

> python main.py -g chess_v4/obj_test_data -v 1

> python main.py -s 1 -v 1

> python main.py -g chess_v4/obj_test_data

> python main.py -s 1

> python main.py -v 1

> python main_interactive.py -g chess_v4/obj_test_data -s 1 -v 1

> python main_interactive.py -g chess_v4/obj_test_data -s 1

> python main_interactive.py -g chess_v4/obj_test_data -v 1

> python main_interactive.py -s 1 -v 1

> python main_interactive.py -g chess_v4/obj_test_data

> python main_interactive.py -s 1

> python main_interactive.py -v 1
