import cv2
from ultralytics import solutions


def default_line_region(width, height, y_frac=0.6, x_margin_frac=0.08):
    y = int(height * y_frac)
    x0 = int(width * x_margin_frac)
    x1 = int(width * (1 - x_margin_frac))
    return [(x0, y), (x1, y)]


class PeopleCounter:
    def __init__(self, region, model="yolo26n.pt", tracker="bytetrack.yaml", classes=(0,)):
        self.solution = solutions.ObjectCounter(
            model=model,
            region=region,
            classes=list(classes),
            tracker=tracker,
            show=False,
        )

    def process_video(self, input_path, output_path, display=False):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open writer for {output_path}")

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                results = self.solution(frame)
                writer.write(results.plot_im)

                if display:
                    cv2.imshow("People Counter", results.plot_im)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            writer.release()
            if display:
                cv2.destroyAllWindows()

        return {
            "in_count": self.solution.in_count,
            "out_count": self.solution.out_count,
            "classwise_count": dict(self.solution.classwise_count),
        }
