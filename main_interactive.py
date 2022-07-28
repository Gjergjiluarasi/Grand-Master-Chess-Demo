from utils import *
from multiprocessing import Process, Value

def main():

    """ Main program """
    # Pause process
    pause_flag =  Value('d', 0)
    pause_ctr = Value('d', 0)
    pauseProcess = Process(target=pause_game, args=(pause_flag, pause_ctr, ))
    pauseProcess.start()

    # Argument parser
    parser = argparse.ArgumentParser(description='Grand Master Chess')
    parser.add_argument("-s", "--save_images", help="Saves images during game run")
    parser.add_argument("-v", "--verbose", help="Show results during game run")
    parser.add_argument("-g", "--game_path", help="Set custom game path of source images")
    args = parser.parse_args()
    save_images = int(args.save_images) if args.save_images != None else 0
    show = int(args.verbose) if args.verbose != None else 0
    game_path = args.game_path if args.game_path != None else './full_game_test'
    print("\n---------------------------------------------------------------------------------------------------\n",
            "\t\tGRAND MASTER CHESS\n",
            "\tYou chose save_images =", save_images, "and show =", show,'\n',
            "\tPlaying game in the folder game_path =", game_path, '\n')

    """ Yolo darknet initialization """
    # Get model and class names
    net, className = yolo_init()

    """ Detect chess board tiles """
    # Initialize the dict of chess board tiles
    chess_board_tiles_dict = chess_board_tiles_init()
    # Extracting path of individual image stored in the game directory
    images = glob.glob(game_path+'/*.jpg')
    images = sorted(images)
    # Try to get a first version of the chess board tiles
    for fname in images:
        img = cv2.imread(fname)
        ret = set_chess_board_tiles(img, chess_board_tiles_dict, 1, save_images=save_images, show=0)
        if ret == True:
            break
    # Print the first detected chess board with tiles
    if show == 1:
        print('Initial chess_board_tiles_dict = {')
        for key in chess_board_tiles_dict:
            print(chess_board_tiles_dict[key],',')
        print('}\n')

    """ Start game """
    if ret == True:
        # Read the images and update each time the dict
        frame_durations = []        #"""0"""
        hand_durations = []         #"""1"""
        tile_durations = []         #"""2"""
        img_prc_durations = []      #"""3"""
        yolo_durations = []         #"""4"""
        piece_durations = []        #"""5"""
        twin_durations = []         #"""6"""
        try:
            # Create hand detection mediapipe model
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.1, model_complexity=1, min_tracking_confidence=0.05) as hands:
                frame_start = 0
                game_start = time()
                game_duration = None
                for i, fname in enumerate(images):
                    # Pause game if pause_flag is 1
                    while pause_flag.value == 1:
                        pass
 
                    """0"""
                    frame_start = time()
                    print("\n---------------------------------------------------------------------------------------------------\n\tFRAME "+str(i+1)+"\t\tSOURCE IMAGE: "+fname)
                    # Read image
                    img = cv2.imread(fname)

                    """1"""
                    """ Filter out frames with hands"""
                    hand_start = time()
                    # Convert the BGR image to RGB before processing
                    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    # If there is hand continue to the next image
                    if results.multi_hand_landmarks:
                        # Draw hand landmarks on the image
                        if save_images == 1:
                            annotated_image = img.copy()
                            for hand_landmarks in results.multi_hand_landmarks:
                                print('hand_landmarks:', hand_landmarks)
                                print('Index finger tip coordinates: (',
                                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * img.shape[1]}, '
                                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * img.shape[0]})')
                                mp_drawing.draw_landmarks(annotated_image, hand_landmarks,mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                                cv2.putText(annotated_image, str(i+1), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                cv2.imwrite("output_images/Hands_detected.jpg", annotated_image)
                                cv2.imwrite("output_images/Current_frame.jpg", annotated_image)
                                if show == 1:
                                    cv2.imshow("Hands detected in current frame", annotated_image)
                                    cv2.waitKey()
                        hand_durations.append(time()-hand_start)    
                        print('(i)\t""" Filter out frames with hands""" lasted {:.3f} s \tHandedness: {}'.format(hand_durations[-1], results.multi_handedness))

                        """2"""
                        tile_durations.append(0)
                        print('(ii)\t""" Set chess board tiles """ lasted {:.3f} s'.format(tile_durations[-1]))

                        """3"""
                        img_prc_durations.append(0)
                        print('(iii)\t""" Image loading and preprocessing """ lasted {:.3f} s'.format(img_prc_durations[-1]))

                        """4"""
                        yolo_durations.append(0)
                        print('(iv)\t""" Yolo darknet inference """ lasted {:.3f} s'.format(yolo_durations[-1]))

                        """5"""
                        piece_durations.append(0)
                        print('(v)\t""" Chess piece list generation """ lasted {:.3f} s'.format(piece_durations[-1]))

                        """6"""
                        twin_durations.append(0)
                        print('(vi)\t""" Display digital twin chess board """ lasted {:.3f} s'.format(twin_durations[-1]))

                        """0"""
                        frame_duration = time()-frame_start
                        frame_durations.append(frame_duration)
                        print('(vii)\t""" FRAME {} """" lasted {:.3f} s/ Frequency = {:.3f} Hz'.format(i+1, frame_durations[-1], 1/frame_durations[-1]))

                    else:
                        hand_durations.append(time()-hand_start)    
                        print('(i)\t""" Filter out frames with hands""" lasted {:.3f} s \tHandedness: {}'.format(hand_durations[-1], results.multi_handedness))
        
                        """2"""
                        """ Set chess board tiles """
                        tile_start = time()
                        # Every 20 frames update the board
                        if i%20 == 0:
                            # Update chess_board_tiles_dict with newly detected tiles
                            ret = set_chess_board_tiles(img, chess_board_tiles_dict, 1, save_images=save_images, show=0)
                        # Create a chess board square Polygon from its vertices
                        chess_board_vertices =np.array([[chess_board_tiles_dict['A1'].bounds[0], chess_board_tiles_dict['A1'].bounds[3]],
                                                        [chess_board_tiles_dict['A8'].bounds[0], chess_board_tiles_dict['A8'].bounds[1]],
                                                        [chess_board_tiles_dict['H8'].bounds[2], chess_board_tiles_dict['H8'].bounds[1]],
                                                        [chess_board_tiles_dict['H1'].bounds[2], chess_board_tiles_dict['H1'].bounds[3]]], dtype = np.int32)
                        chess_board_square = Polygon(chess_board_vertices)                         
                        img_with_tiles = img
                        tile_durations.append(time()-tile_start)
                        print('(ii)\t""" Set chess board tiles """ lasted {:.3f} s'.format(tile_durations[-1]))
                        if show == 1:
                            print('\n\nChess board square Polygon: ', chess_board_square) 

                        """3"""
                        """ Image loading and preprocessing """
                        img_prc_start = time()
                        # Read image
                        img_path = fname
                        img = cv2.imread(img_path)
                        # Image preprocessing
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (whT, whT), swapRB=True, crop=False)
                        net.setInput(blob)
                        # Get outputnames for inference
                        layernames = net.getLayerNames()
                        outputnames = [(layernames[int(i - 1)]) for i in net.getUnconnectedOutLayers()]
                        img_prc_durations.append(time()-img_prc_start)
                        print('(iii)\t""" Image loading and preprocessing """ lasted {:.3f} s'.format(img_prc_durations[-1]))

                        """4"""
                        """ Yolo darknet inference """
                        yolo_start = time()
                        # Yolo darknet inference
                        outputs = net.forward(outputnames)
                        yolo_durations.append(time()-yolo_start)
                        print('(iv)\t""" Yolo darknet inference """ lasted {:.3f} s'.format(yolo_durations[-1]))

                        """5"""
                        """ Chess piece list generation """
                        # Generate list of chess pieces
                        piece_start = time()
                        chess_pieces = get_chess_pieces(outputs, img_with_tiles, chess_board_tiles_dict, chess_board_square, className, frame_id=i+1, save_images=save_images, show=show)
                        piece_durations.append(time()-piece_start)
                        print('(v)\t""" Chess piece list generation """ lasted {:.3f} s'.format(piece_durations[-1]))

                        """6"""
                        """ Display digital twin chess board """
                        # Get chess board as formatted string for chessboard module
                        twin_start = time()
                        chess_board_string = get_chess_board(chess_pieces)
                        update_chess_board(chess_board_string,i+1)
                        twin_durations.append(time()-twin_start)
                        print('(vi)\t""" Display digital twin chess board """ lasted {:.3f} s'.format(twin_durations[-1]))
                    
                        """0"""
                        frame_duration = time()-frame_start
                        frame_durations.append(frame_duration)
                        print('(vii)\t""" FRAME {} """" lasted {:.3f} s/ Frequency = {:.3f} Hz'.format(i+1, frame_durations[-1], 1/frame_durations[-1]))

                game_duration = time()-game_start

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            print("\n\tGame interrupted!\n")

        finally:
            # try:
            if game_duration == None:
                game_duration = time() - game_start if game_start != None else 0
            """7"""
            """ Show program statistics"""
            cv2.destroyAllWindows()
            # Calculate expected values
            avg_frame_durations = np.mean(frame_durations)       #"""0"""
            avg_hand_durations = np.mean(hand_durations)         #"""1"""
            avg_tile_durations = np.mean(tile_durations)         #"""2"""
            avg_img_prc_durations = np.mean(img_prc_durations)   #"""3"""
            avg_yolo_durations = np.mean(yolo_durations)         #"""4"""
            avg_piece_durations = np.mean(piece_durations)       #"""5"""
            avg_twin_durations = np.mean(twin_durations)         #"""6"""
            # Print expected values
            print("\n---------------------------------------------------------------------------------------------------\n\tPROGRAM STATISTICS:")
            print("Total game duration: {:.3f} s".format(game_duration))
            print("Average frame durations(average sample frequency): {:.3f} s ({:.3f} Hz)".format(avg_frame_durations, 1/avg_frame_durations))
            print("Min frame duration (max sample frequency): {:.3f} s ({:.3f} Hz)".format(np.min(frame_durations), 1/np.min(frame_durations)))
            print("Max frame duration (min sample frequency): {:.3f} s ({:.3f} Hz)".format(np.max(frame_durations), 1/np.max(frame_durations)))
            print("\n---------------------------------------------------------------------------------------------------\n")
            print("\tPlease press Ctrl+C to exit program!")
            
            # Pie chart of statistics
            avg_total_durations = avg_hand_durations + avg_tile_durations + avg_img_prc_durations + avg_yolo_durations + avg_piece_durations + avg_twin_durations
            labels = 'mediapipe inference', 'chess board tiles', 'image preprocessing', 'yolo inference', 'chess piece', 'digital twin'
            sizes = np.array([avg_hand_durations, avg_tile_durations, avg_img_prc_durations, avg_yolo_durations, avg_piece_durations, avg_twin_durations])*100/avg_total_durations
            explode = (0.1, 0, 0, 0.1, 0, 0)  # only "explode" the 3rd slice (i.e. 'yolo inference')
            
            # Plot statistics
            fig, ax = plt.subplots(1,2)
            # Plot of Frame Durations and Yolo Inference Durations
            fig.text(0.1, 0.12,"PROGRAM STATISTICS:", style='normal')
            fig.text(0.05, 0.10, "Total game duration: {:.3f} s".format(game_duration), style='oblique')
            fig.text(0.05, 0.08, "Min frame duration (max sample frequency): {:.3f} s ({:.3f} Hz)".format(np.min(frame_durations), 1/np.min(frame_durations)), style='oblique')
            fig.text(0.05, 0.06, "Max frame duration (min sample frequency): {:.3f} s ({:.3f} Hz)".format(np.max(frame_durations), 1/np.max(frame_durations)), style='oblique')
            fig.text(0.05, 0.04,"Average frame durations(average sample frequency): {:.3f} s ({:.3f} Hz)".format(avg_frame_durations, 1/avg_frame_durations), style='oblique')
            yolo_durations = np.array(yolo_durations)
            fig.text(0.05, 0.02, "Average yolo inference duration: {:.3f} s".format(np.mean(yolo_durations[yolo_durations!=0])), style='oblique')
            hand_durations = np.array(hand_durations)
            fig.text(0.05, 0.00, "Average mediapipe inference duration: {:.3f} s".format(np.mean(hand_durations[hand_durations!=0])), style='oblique')
            ax[0].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
            ax[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            end_frame = len(frame_durations) if len(frame_durations)<201 else 200
            ax[1].plot(range(1,len(frame_durations[:end_frame])+1),frame_durations[:end_frame], label = 'frame durations')
            ax[1].plot(range(1,len(yolo_durations[:end_frame])+1),yolo_durations[:end_frame], label = 'yolo inference durations')
            ax[1].plot(range(1,len(hand_durations[:end_frame])+1),hand_durations[:end_frame], label = 'mediapipe hand inference durations')
            ax[1].plot(range(1,len(tile_durations[:end_frame])+1),tile_durations[:end_frame], label = 'chess board tiles durations')
            ax[1].plot(range(1,len(piece_durations[:end_frame])+1),piece_durations[:end_frame], label = 'chess piece durations')
            ax[1].set_title("Frame, Mediapipe and Yolo Inference Durations in seconds")
            ax[1].set_xlabel("frame_id [i]")
            ax[1].set_ylabel("$\mathregular{durations_i}$ [s]")
            ax[1].legend()
            ax[1].grid()
            figManager = plt.get_current_fig_manager()
            figManager.full_screen_toggle()
            plt.show()
            fig.set_size_inches(13.6, 7.65)
            fig.savefig("output_images/Program_statistics.jpg", dpi = 800)
            # except:
            #     print("\n\tCould not plot statistics! Leaving program...\n")

    else:
        cv2.destroyAllWindows()
        print("\n\tCould not detect chess board! Leaving program...\n")
    
    pauseProcess.join()

    return 0

if __name__ == "__main__":
    main()
