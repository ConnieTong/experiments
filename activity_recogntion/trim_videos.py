import json
import subprocess
from os import listdir
from os.path import isfile, join


def trim(input_file, start_time, duration, output_file):
    command = ['ffmpeg',
               '-i', '"%s"' % input_file,
               '-ss', str(start_time),
               '-t', str(duration),
               '-c:v', 'libx264', '-c:a', 'copy',
               '-threads', '1',
               '-loglevel', 'panic',
               '"%s"' % output_file]
    subprocess.call(" ".join(command), shell=True)


def check():
    ready = {}
    done = []
    for g in listdir("extracted_videos/"):
        key = "_".join(g.split(".mp4")[0].split("_")[1: -1])
        if key not in ready:
            ready[key] = 1
        else:
            ready[key] += 1
        assert key in anno_data
    for w in ready:
        if ready[w] == len(anno_data[w]["annotations"]):
            done.append(w)
    return done


if __name__ == '__main__':
    file_name = "activity_net.v1-3.min.json"
    with open(file_name) as f:
        data = json.load(f)
    anno_data = data["database"]
    done_videos = check()
    duplicate_names = [d for d in listdir("extracted_videos/")]

    mypath = "Housework Activities/"
    onlylabels = [f for f in listdir(mypath)]
    all_videos = []
    skipped = 0
    for l in onlylabels:
        onlyfiles = [join(mypath + l, f) for f in listdir(mypath + l) if isfile(join(mypath + l, f))]
        all_videos.extend(onlyfiles)
    for video_path in all_videos:
        video_name = video_path.split("/")[-1]
        video_key = video_name.split(".mp4")[0]
        if video_key in done_videos:
            skipped += 1
            print("skipped", skipped, video_key)
            continue
        if video_key in anno_data:
            print("Processing %s into %d clips" % (video_path, len(anno_data[video_key]["annotations"])))
            count = 0
            for ann in anno_data[video_key]["annotations"]:
                out_name = "extracted_videos/%s_%s_%d.mp4" % (ann["label"], video_key, count)
                if out_name in duplicate_names:
                    out_name = "extracted_videos/%s_%s_%d_.mp4" % (ann["label"], video_key+"dup", count)
                print(" ", video_path, ann["segment"][0], ann["segment"][1] - ann["segment"][0], out_name)
                trim(video_path, ann["segment"][0], ann["segment"][1] - ann["segment"][0], out_name)
                count += 1
                duplicate_names.append(out_name)
