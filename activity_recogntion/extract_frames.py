import subprocess
import os


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def extract_by_fps(input_file, output_file):
    command = ['ffmpeg',
               '-i', '"%s"' % input_file,
               "-vf", "fps=6",
               '"%s"' % output_file,
               "-hide_banner"]
    subprocess.call(" ".join(command), shell=True)


if __name__ == '__main__':
    all_videos = os.listdir("extracted_videos/")
    videos_done = 0
    for v in all_videos:
        path_to_store_frames = "frames/%s" % v
        if not os.path.exists(path_to_store_frames):
            os.makedirs(path_to_store_frames)
        extract_by_fps("extracted_videos/%s" % v, "%s/img%%07d.jpg" % path_to_store_frames)
        videos_done += 1
        print("%s%sDone %d/%d, frames are stored in %s%s" % (bcolors.OKGREEN, bcolors.BOLD, videos_done, 
                                                             len(all_videos), path_to_store_frames, bcolors.ENDC))
