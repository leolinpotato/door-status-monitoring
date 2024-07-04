import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from ultralytics import YOLO
import ipdb
from scipy.ndimage import gaussian_filter1d

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='../Tests', help='Name of the directory to process')
parser.add_argument('--output', type=str, default='output.json', help='Name of the output file')
args = parser.parse_args()

model = YOLO("yolov8l.pt")

def guess_door(directory, video_path):
    full_path = f"{directory}/{video_path}"

    # parameters
    mask_area_ratio = 0.8
    pixel_value_threshold = 10 # sum the number of pixel whose value over it in the detected area
    ratio_threshold_ratio = 0.4 # only the frame with ratio over it will be considered open/close
    frame_sliding_window = 20 # if the number(this) of continuos frames are over ratio threshold
    window_ratio = 0.8 # if in frame_sliding_window, there are window_ratio frames over the ratio_threshold
    pixel_avg_ratio = 5
    sigma = 6  # sigma for 1D gaussian filter

    def divide_list(lst):
        result = []
        sublist = [lst[0]]

        for i in range(1, len(lst)):
            if lst[i]-lst[i-1] == 1:
                sublist.append(lst[i])
            else:
                result.append(sublist)
                sublist = [lst[i]]

        result.append(sublist)
        return result

    def guess(lst):
        if len(lst) > 40:
            return lst[20]
        return lst[int(len(lst)*0.55)]
    
    def normalize(data, upper=97, lower=5):
        data_array = np.array(data)
        # Calculate the specified percentile value
        upper_bound = np.percentile(data_array, upper)
        lower_bound = np.percentile(data_array, lower)
        # Clip the values between upper bound and lower bound
        clipped_data = np.clip(data_array, lower_bound, upper_bound)
        return clipped_data.tolist()

    cap = cv2.VideoCapture(full_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_count = 0
    
    # *****************************************************
    # *** First pass -> Get average value in each pixel *** 
    # *****************************************************
    pixel_sum = np.zeros((frame_height, frame_width), dtype=np.uint32)

    ret, frame_prev = cap.read()
    frame_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    frame_prev = cv2.GaussianBlur(frame_prev, (21, 21), 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (21, 21), 0) 

        # use frame by frame pixel difference to capture moving objects (people, door, and window)
        frame_diff = cv2.absdiff(frame_prev, frame)
        frame_prev = frame.copy()
        frame = frame_diff 

        pixel_sum += frame

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # use pixel_avg as mask to filter out window
    pixel_avg = pixel_sum/frame_count
    pixel_avg = pixel_avg.astype(np.uint8)*pixel_avg_ratio  


    # setup mask area at middle
    y1_masked = int((1-mask_area_ratio)*frame_height)
    y2_masked = int(mask_area_ratio*frame_height)
    x1_masked = int((1-mask_area_ratio)*frame_width)
    x2_masked = int(mask_area_ratio*frame_width)

    mask = np.zeros((y2_masked-y1_masked, x2_masked-x1_masked), dtype='uint8')

    ratios = []

    # *****************************************************************************
    # *** Second pass -> filter out people and window to capture door movements *** 
    # *****************************************************************************
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

    ret, frame_prev = cap.read()
    frame_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    frame_prev = cv2.GaussianBlur(frame_prev, (21, 21), 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # use yolo to find out "0: people, 24: backpack, 26: handbag, 28: suitcase"
        results = model.predict(source=frame, classes=[0, 24, 26, 28], conf=0.15, show=False)  # results is a list of length 1
        boxes = results[0].boxes.cpu().numpy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.equalizeHist(frame)
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
        
        # frame = bg_subtractor.apply(frame)
        frame_diff = cv2.absdiff(frame_prev, frame)
        frame_prev = frame.copy()
        frame = frame_diff

        # mask out middle area
        frame[y1_masked:y2_masked, x1_masked:x2_masked] = mask

        # mask out predicted objects
        for xyxy in boxes.xyxy:
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 0), -1)
        
        # subtract pixel_avg from frame, try to filter out window
        frame = np.clip(frame.astype(np.int16)-pixel_avg.astype(np.int16), 0, 255).astype(np.uint8)

        # calculate average pixel value of current frame
        area = frame[:, :]
        ratio = np.sum(area) / area.size
        ratios.append(ratio)

        frame = cv2.putText(frame, f"Frame: {frame_count}", (frame_height//2, frame_width//2), cv2.FONT_HERSHEY_SIMPLEX
    , 1, (255, 255, 255), 2)
        
        # cv2.imshow('Frame', frame)
        # print(f"Frame number: {frame_count}, Ratio: {ratio}%")
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ************************************
    # *** process ratios list and plot *** 
    # ************************************
    ratios.pop(0)
    ratios = normalize(ratios)
    ratios = gaussian_filter1d(ratios, sigma).tolist()
    max_ratio = max(ratios)
    min_ratio = min(ratios)
    ratios.insert(0, ratios[0])
    ratios.insert(0, ratios[0])
    max_ratio = max(ratios)
    ratio_threshold = min_ratio + (max_ratio - min_ratio) * ratio_threshold_ratio 
    # plt.plot(ratios)
    # plt.axhline(y=ratio_threshold , color='r', linestyle='--', label=f'Horizontal line at y = {(2/5):.2f} * max_value')
    # plt.title('ratio in frames')
    # plt.xlabel('frame')
    # plt.ylabel('ratio(pixel value)')
    # plt.savefig(f"./Figs/yolo_normalized_{fig_path}")
    # plt.clf()

    # ***********************************
    # *** guess open and close frames *** 
    # ***********************************
    candidates_frames =[]

    video_info = {
        "video_filename": video_path,
        "annotations": [
            {
                "object": "Door",
                "states": []
            }
        ]
    }

    for start in range(len(ratios)-frame_sliding_window):
        if len([x for x in range(start, start+frame_sliding_window) if ratios[x] > ratio_threshold]) > frame_sliding_window * window_ratio:
            candidates_frames.append(start)
    
    if candidates_frames:
        # divide candidates_frames into clusters
        candidates_frames_divided = divide_list(candidates_frames)
        for i, lst in enumerate(candidates_frames_divided):
            guessed_frame = guess(lst)
            if i % 2 == 0:
                status = "Opening"
            else:
                status = "Closing"
            video_info["annotations"][0]["states"].append({
                "state_id": i+1,
                "description": status,
                "guessed_frame": guessed_frame
            })
            print(f"Section {i+1} guessed frame {guessed_frame}")

    cap.release()
    cv2.destroyAllWindows()
    return video_info

def scan_videos(directory):
    """Scan the specified directory for MP4 files and generate JSON annotations."""
    video_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
    videos_info = []

    for video_file in video_files:
        videos_info.append(guess_door(directory, video_file))

    return videos_info

def generate_json(output_filename, videos_info):
    """Generate a JSON file with the provided video information."""
    with open(output_filename, 'w') as file:
        json.dump({"videos": videos_info}, file, indent=4)

def main():
    directory = args.dir # Specify the directory to scan
    output_filename = args.output  # Output JSON file name
    videos_info = scan_videos(directory)
    generate_json(output_filename, videos_info)
    print(f"Generated JSON file '{output_filename}' with video annotations.")

if __name__ == "__main__":
    main()
