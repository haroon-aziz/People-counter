import argparse
import sys
import json
import cv2

sys.path.insert(0, "..")
from peoplecounter.counter import PeopleCounter, default_line_region


def parse_point_pair(value):
    x, y = value.split(",")
    return (int(x), int(y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="people_counted.mp4")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--line-start", type=parse_point_pair, default=None, help="x,y")
    parser.add_argument("--line-end", type=parse_point_pair, default=None, help="x,y")
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    if args.line_start and args.line_end:
        region = [args.line_start, args.line_end]
    else:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {args.input}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        region = default_line_region(width, height)

    counter = PeopleCounter(region=region, model=args.model, tracker=args.tracker, classes=(0,))
    result = counter.process_video(args.input, args.output, display=args.display)

    print(json.dumps(result, indent=2))
    print(f"Wrote annotated video to {args.output}")


if __name__ == "__main__":
    main()
