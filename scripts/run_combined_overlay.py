from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


COCO_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COURT_POLYGON_CLASS_NAMES = [
    "court_bottom_left_far",
    "court_bottom_right_far",
    "court_bottom_right_close",
    "court_bottom_left_close",
]

COURT_KEYPOINT_LABELS = [
    "court_bottom_left_close",
    "court_bottom_left_far",
    "court_bottom_right_close",
    "court_bottom_right_far",
    "court_top_left_close",
    "court_top_left_far",
    "court_top_right_close",
    "court_top_right_far",
    "net_bottom_left",
    "net_bottom_right",
    "net_top_left",
    "net_top_right",
    "service_centre_close",
    "service_centre_far",
    "service_left_close",
    "service_left_far",
    "service_right_close",
    "service_right_far",
]

PLAYER_BOX_COLOR = (60, 200, 60)
POSE_COLOR = (255, 220, 0)
BALL_COLOR = (30, 30, 255)
COURT_COLOR = (255, 140, 0)
TEXT_COLOR = (255, 255, 255)


@dataclass
class TrackMemory:
    box: np.ndarray
    conf: float
    age: int = 1


@dataclass
class BoxAnnotation:
    label: str
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    group_id: int | None = None


@dataclass
class PointAnnotation:
    label: str
    x: float
    y: float
    group_id: int | None = None


@dataclass
class FrameAnnotations:
    name: str
    width: int
    height: int
    boxes: list[BoxAnnotation] = field(default_factory=list)
    points: list[PointAnnotation] = field(default_factory=list)


@dataclass
class VideoJob:
    video_path: Path
    output_path: Path
    xml_path: Path | None
    cvat_frames_dir: Path | None


@dataclass
class BallTrackState:
    last_trusted_center: tuple[float, float] | None = None
    gap_frames: int = 0
    reject_streak: int = 0
    center_history: deque[tuple[float, float]] = field(default_factory=deque)
    size_history: deque[tuple[float, float]] = field(default_factory=deque)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the trained padel models together and save overlay video plus optional CVAT XML.",
    )
    parser.add_argument("--video", help="Input video path.")
    parser.add_argument("--video-dir", help="Optional directory of input videos for batch inference.")
    parser.add_argument(
        "--video-pattern",
        default="*.mp4",
        help="Glob used with --video-dir. Default: *.mp4",
    )
    parser.add_argument(
        "--players-detection-model",
        default="runs/detect/runs/train/zenodo_players_detection_baseline3/weights/best.pt",
        help="Players detection checkpoint.",
    )
    parser.add_argument(
        "--players-pose-model",
        default="runs/pose/runs/train/zenodo_players_pose_finish_from_baseline/weights/best.pt",
        help="Players pose checkpoint.",
    )
    parser.add_argument(
        "--ball-model",
        default="runs/detect/runs/train/zenodo_ball_detection_baseline/weights/best.pt",
        help="Ball detection checkpoint.",
    )
    parser.add_argument(
        "--court-model",
        default="runs/pose/runs/train/court_keypoints_no_cage_v1/weights/best.pt",
        help="Court keypoints checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/overlays/combined_overlay.mp4",
        help="Output overlay video path for single-video mode.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for batch mode. Each clip is saved as <stem>.mp4.",
    )
    parser.add_argument(
        "--xml-output",
        default=None,
        help="Optional CVAT XML output path for single-video mode. Defaults to the overlay path with .xml suffix.",
    )
    parser.add_argument(
        "--xml-dir",
        default=None,
        help="Optional CVAT XML output directory for batch mode. Defaults to --output-dir.",
    )
    parser.add_argument(
        "--cvat-frames-dir",
        default=None,
        help=(
            "Optional frame export directory for CVAT image-task import. "
            "Single-video mode writes frames into this directory; batch mode writes one subdirectory per clip."
        ),
    )
    parser.add_argument("--device", default="0", help='Model device, e.g. "0" or "cpu".')
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference size for the ball model.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap for quick tests.")
    parser.add_argument("--players-conf", type=float, default=0.25)
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--ball-conf", type=float, default=0.10)
    parser.add_argument("--court-conf", type=float, default=0.15)
    parser.add_argument(
        "--max-players",
        type=int,
        default=0,
        help="Optional hard cap for kept player detections. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-balls",
        type=int,
        default=0,
        help="Optional hard cap for kept ball detections. Use 0 to disable.",
    )
    parser.add_argument(
        "--court-seed-frames",
        type=int,
        default=5,
        help="Number of initial frames used to auto-seed the court polygon before falling back.",
    )
    parser.add_argument(
        "--ball-smoothing-window",
        type=int,
        default=5,
        help="Rolling window used to smooth accepted ball centers.",
    )
    parser.add_argument(
        "--ball-max-step-ratio",
        type=float,
        default=0.09,
        help="Maximum normalized inter-frame ball jump before rejection.",
    )
    parser.add_argument(
        "--ball-reacquire-gap-frames",
        type=int,
        default=10,
        help="Gap length after which a new distant ball candidate can be trusted again.",
    )
    parser.add_argument(
        "--ball-max-reject-streak",
        type=int,
        default=5,
        help="Consecutive rejected ball frames before re-locking to a new candidate.",
    )
    return parser


def require_file(path_str: str, label: str) -> Path:
    path = Path(path_str).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def validate_args(args: argparse.Namespace) -> None:
    if bool(args.video) == bool(args.video_dir):
        raise ValueError("Provide exactly one of --video or --video-dir.")
    if args.video_dir and not args.output_dir:
        raise ValueError("--output-dir is required when using --video-dir.")


def collect_video_jobs(args: argparse.Namespace) -> list[VideoJob]:
    jobs: list[VideoJob] = []

    if args.video:
        video_path = require_file(args.video, "Video")
        output_path = Path(args.output).resolve()
        xml_path = Path(args.xml_output).resolve() if args.xml_output else output_path.with_suffix(".xml")
        cvat_frames_dir = Path(args.cvat_frames_dir).resolve() if args.cvat_frames_dir else None
        jobs.append(
            VideoJob(
                video_path=video_path,
                output_path=output_path,
                xml_path=xml_path,
                cvat_frames_dir=cvat_frames_dir,
            )
        )
        return jobs

    video_dir = Path(args.video_dir).resolve()
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    output_dir = Path(args.output_dir).resolve()
    xml_dir = Path(args.xml_dir).resolve() if args.xml_dir else output_dir
    base_frames_dir = Path(args.cvat_frames_dir).resolve() if args.cvat_frames_dir else None

    for video_path in sorted(video_dir.glob(args.video_pattern)):
        if not video_path.is_file():
            continue
        jobs.append(
            VideoJob(
                video_path=video_path,
                output_path=output_dir / f"{video_path.stem}.mp4",
                xml_path=xml_dir / f"{video_path.stem}.xml",
                cvat_frames_dir=base_frames_dir / video_path.stem if base_frames_dir else None,
            )
        )

    if not jobs:
        raise FileNotFoundError(f"No videos matched {args.video_pattern!r} in {video_dir}")

    return jobs


def draw_label(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def result_boxes_to_numpy(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if result.boxes is None:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)
    xyxy = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, "cpu") else result.boxes.xyxy
    conf = result.boxes.conf.cpu().numpy() if hasattr(result.boxes.conf, "cpu") else result.boxes.conf
    cls = result.boxes.cls.cpu().numpy().astype(int) if hasattr(result.boxes.cls, "cpu") else result.boxes.cls
    return xyxy, conf, cls


def result_keypoints_to_numpy(result) -> np.ndarray:
    if result.keypoints is None:
        return np.empty((0, 0, 2))
    return result.keypoints.xy.cpu().numpy() if hasattr(result.keypoints.xy, "cpu") else result.keypoints.xy


def extract_court_points_by_name(result) -> dict[str, tuple[float, float]]:
    if result.boxes is None or result.keypoints is None:
        return {}

    class_ids = result.boxes.cls.cpu().numpy().astype(int) if hasattr(result.boxes.cls, "cpu") else result.boxes.cls
    keypoints = result_keypoints_to_numpy(result)
    names = result.names
    points_by_name: dict[str, tuple[float, float]] = {}

    for class_id, point_group in zip(class_ids, keypoints):
        if len(point_group) == 0:
            continue
        point = point_group[0]
        if np.any(point <= 0):
            continue
        class_name = str(names.get(int(class_id), class_id))
        points_by_name[class_name] = (float(point[0]), float(point[1]))

    return points_by_name


def build_court_polygon_from_points(points_by_name: dict[str, tuple[float, float]]) -> np.ndarray | None:
    if not all(class_name in points_by_name for class_name in COURT_POLYGON_CLASS_NAMES):
        return None
    return np.array([points_by_name[class_name] for class_name in COURT_POLYGON_CLASS_NAMES], dtype=np.float32)


def build_court_polygon(result) -> np.ndarray | None:
    return build_court_polygon_from_points(extract_court_points_by_name(result))


def default_court_polygon(frame_width: int, frame_height: int) -> np.ndarray:
    width = float(frame_width)
    height = float(frame_height)
    return np.array(
        [
            (0.20 * width, 0.30 * height),
            (0.80 * width, 0.30 * height),
            (0.95 * width, 0.95 * height),
            (0.05 * width, 0.95 * height),
        ],
        dtype=np.float32,
    )


def point_inside_polygon(point: tuple[float, float], polygon: np.ndarray | None) -> bool:
    if polygon is None:
        return True
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def box_center(box: np.ndarray) -> tuple[float, float]:
    return float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)


def box_feet(box: np.ndarray) -> tuple[float, float]:
    return float((box[0] + box[2]) / 2.0), float(box[3])


def box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    inter_x1 = max(float(box_a[0]), float(box_b[0]))
    inter_y1 = max(float(box_a[1]), float(box_b[1]))
    inter_x2 = min(float(box_a[2]), float(box_b[2]))
    inter_y2 = min(float(box_a[3]), float(box_b[3]))
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union = box_area(box_a) + box_area(box_b) - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def center_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    center_a = box_center(box_a)
    center_b = box_center(box_b)
    return float(np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]))


def candidate_score(
    box: np.ndarray,
    conf: float,
    polygon: np.ndarray | None,
    point_selector=box_center,
) -> float:
    score = float(conf)
    if point_inside_polygon(point_selector(box), polygon):
        score += 0.25
    return score


def pose_box_matches_player_box(pose_box: np.ndarray, player_box: np.ndarray) -> bool:
    center_x, center_y = box_center(pose_box)
    if (
        player_box[0] <= center_x <= player_box[2]
        and player_box[1] <= center_y <= player_box[3]
    ):
        return True
    return box_iou(pose_box, player_box) >= 0.25


def filter_pose_indices_by_player_boxes(
    pose_result,
    pose_indices: list[int],
    pose_memories: list[TrackMemory],
    players_detection_result,
    player_indices: list[int],
) -> tuple[list[int], list[TrackMemory]]:
    player_boxes, _, _ = result_boxes_to_numpy(players_detection_result)
    pose_boxes, _, _ = result_boxes_to_numpy(pose_result)
    if len(player_indices) == 0 or len(pose_indices) == 0:
        return [], []

    kept_indices: list[int] = []
    kept_memories: list[TrackMemory] = []
    for index, memory in zip(pose_indices, pose_memories):
        pose_box = pose_boxes[index]
        if any(pose_box_matches_player_box(pose_box, player_boxes[player_index]) for player_index in player_indices):
            kept_indices.append(index)
            kept_memories.append(memory)

    return kept_indices, kept_memories


def select_stable_indices(
    result,
    polygon: np.ndarray | None,
    memories: list[TrackMemory],
    *,
    max_count: int,
    match_distance_ratio: float,
    min_match_iou: float,
    new_track_conf: float,
    prefer_existing_bonus: float,
    frame_width: int,
    frame_height: int,
    point_selector=box_center,
) -> tuple[list[int], list[TrackMemory]]:
    boxes, confs, _ = result_boxes_to_numpy(result)
    if len(boxes) == 0:
        return [], []

    frame_diag = float(np.hypot(frame_width, frame_height))
    max_match_distance = match_distance_ratio * frame_diag
    unmatched_indices = set(range(len(boxes)))
    selected_indices: list[int] = []
    updated_memories: list[TrackMemory] = []

    for memory in sorted(memories, key=lambda item: (-item.age, -item.conf)):
        best_index = -1
        best_score = float("-inf")
        for index in unmatched_indices:
            distance = center_distance(memory.box, boxes[index])
            iou = box_iou(memory.box, boxes[index])
            if iou < min_match_iou and distance > max_match_distance:
                continue
            score = candidate_score(boxes[index], float(confs[index]), polygon, point_selector) + prefer_existing_bonus
            score += 0.75 * iou
            score -= distance / max(frame_diag, 1.0)
            if score > best_score:
                best_score = score
                best_index = index

        if best_index >= 0:
            unmatched_indices.remove(best_index)
            selected_indices.append(best_index)
            updated_memories.append(
                TrackMemory(
                    box=boxes[best_index].copy(),
                    conf=float(confs[best_index]),
                    age=memory.age + 1,
                )
            )

    for index in sorted(
        unmatched_indices,
        key=lambda item: candidate_score(boxes[item], float(confs[item]), polygon, point_selector),
        reverse=True,
    ):
        if float(confs[index]) < new_track_conf:
            continue
        selected_indices.append(index)
        updated_memories.append(
            TrackMemory(
                box=boxes[index].copy(),
                conf=float(confs[index]),
                age=1,
            )
        )

    if max_count > 0 and len(selected_indices) > max_count:
        ranked_pairs = sorted(
            zip(selected_indices, updated_memories),
            key=lambda item: (-item[1].age, -candidate_score(boxes[item[0]], float(confs[item[0]]), polygon, point_selector)),
        )
        kept_pairs = ranked_pairs[:max_count]
        selected_indices = [index for index, _ in kept_pairs]
        updated_memories = [memory for _, memory in kept_pairs]

    return selected_indices, updated_memories


def select_ball_boxes(
    result,
    selected_indices: list[int],
    polygon: np.ndarray | None,
    state: BallTrackState,
    *,
    frame_width: int,
    frame_height: int,
    max_step_ratio: float,
    smoothing_window: int,
    reacquire_gap_frames: int,
    max_reject_streak: int,
) -> tuple[list[np.ndarray], list[float], BallTrackState]:
    boxes, confs, _ = result_boxes_to_numpy(result)
    if len(selected_indices) == 0:
        state.gap_frames += 1
        return [], [], state

    ranked_indices = sorted(
        selected_indices,
        key=lambda index: candidate_score(boxes[index], float(confs[index]), polygon, box_center),
        reverse=True,
    )
    candidate_index = ranked_indices[0]
    candidate_box = boxes[candidate_index].copy()
    candidate_conf = float(confs[candidate_index])
    candidate_center = box_center(candidate_box)

    diagonal = float(np.hypot(frame_width, frame_height))
    max_step = max_step_ratio * diagonal * max(1, state.gap_frames + 1)

    if state.last_trusted_center is not None:
        distance = float(
            np.hypot(
                candidate_center[0] - state.last_trusted_center[0],
                candidate_center[1] - state.last_trusted_center[1],
            )
        )
        if distance > max_step:
            state.reject_streak += 1
            state.gap_frames += 1
            if state.gap_frames < reacquire_gap_frames and state.reject_streak < max_reject_streak:
                return [], [], state

    width = float(candidate_box[2] - candidate_box[0])
    height = float(candidate_box[3] - candidate_box[1])
    state.center_history.append(candidate_center)
    state.size_history.append((width, height))
    while len(state.center_history) > max(1, smoothing_window):
        state.center_history.popleft()
    while len(state.size_history) > max(1, smoothing_window):
        state.size_history.popleft()

    smoothed_center = tuple(np.mean(np.array(state.center_history, dtype=float), axis=0).tolist())
    smoothed_size = tuple(np.mean(np.array(state.size_history, dtype=float), axis=0).tolist())

    smoothed_box = np.array(
        [
            smoothed_center[0] - smoothed_size[0] / 2.0,
            smoothed_center[1] - smoothed_size[1] / 2.0,
            smoothed_center[0] + smoothed_size[0] / 2.0,
            smoothed_center[1] + smoothed_size[1] / 2.0,
        ],
        dtype=np.float32,
    )
    smoothed_box[0] = max(0.0, min(smoothed_box[0], float(frame_width - 1)))
    smoothed_box[1] = max(0.0, min(smoothed_box[1], float(frame_height - 1)))
    smoothed_box[2] = max(0.0, min(smoothed_box[2], float(frame_width - 1)))
    smoothed_box[3] = max(0.0, min(smoothed_box[3], float(frame_height - 1)))

    state.last_trusted_center = smoothed_center
    state.gap_frames = 0
    state.reject_streak = 0
    return [smoothed_box], [candidate_conf], state


def draw_players_detection(frame: np.ndarray, result, selected_indices: list[int]) -> int:
    if result.boxes is None:
        return 0
    boxes = list(result.boxes)
    for index in selected_indices:
        box = boxes[index]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), PLAYER_BOX_COLOR, 2)
        draw_label(frame, f"player {conf:.2f}", (x1, max(20, y1 - 8)), PLAYER_BOX_COLOR)
    return len(selected_indices)


def draw_players_pose(frame: np.ndarray, result, selected_indices: list[int]) -> int:
    if result.boxes is None or result.keypoints is None:
        return 0

    boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, "cpu") else result.boxes.xyxy
    keypoints = result_keypoints_to_numpy(result)

    for index in selected_indices:
        box_xyxy = boxes[index]
        pose_points = keypoints[index]
        x1, y1, x2, y2 = map(int, box_xyxy.tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), POSE_COLOR, 2)
        for start_idx, end_idx in COCO_SKELETON:
            if start_idx >= len(pose_points) or end_idx >= len(pose_points):
                continue
            start_point = pose_points[start_idx]
            end_point = pose_points[end_idx]
            if np.any(start_point <= 0) or np.any(end_point <= 0):
                continue
            cv2.line(
                frame,
                (int(start_point[0]), int(start_point[1])),
                (int(end_point[0]), int(end_point[1])),
                POSE_COLOR,
                2,
            )
        for point in pose_points:
            if np.any(point <= 0):
                continue
            cv2.circle(frame, (int(point[0]), int(point[1])), 3, POSE_COLOR, -1)
    return len(selected_indices)


def draw_ball_detection(frame: np.ndarray, selected_boxes: list[np.ndarray], selected_confs: list[float]) -> int:
    for box, conf in zip(selected_boxes, selected_confs):
        x1, y1, x2, y2 = map(int, box.tolist())
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        radius = max(4, int(max(x2 - x1, y2 - y1) / 2))
        cv2.circle(frame, (center_x, center_y), radius, BALL_COLOR, 2)
        draw_label(frame, f"ball {conf:.2f}", (x1, max(20, y1 - 8)), BALL_COLOR)
    return len(selected_boxes)


def draw_court_keypoints(frame: np.ndarray, result) -> int:
    count = 0
    if result.boxes is None or result.keypoints is None:
        return count

    class_ids = result.boxes.cls.cpu().numpy().astype(int) if hasattr(result.boxes.cls, "cpu") else result.boxes.cls
    keypoints = result_keypoints_to_numpy(result)
    names = result.names

    for class_id, point_group in zip(class_ids, keypoints):
        if len(point_group) == 0:
            continue
        point = point_group[0]
        if np.any(point <= 0):
            continue
        x_coord, y_coord = int(point[0]), int(point[1])
        cv2.circle(frame, (x_coord, y_coord), 5, COURT_COLOR, -1)
        draw_label(frame, str(names.get(class_id, class_id)), (x_coord + 6, y_coord - 6), COURT_COLOR)
        count += 1
    return count


def choose_seed_court_polygon(
    capture: cv2.VideoCapture,
    court_model: YOLO,
    *,
    court_conf: float,
    device: str,
    seed_frames: int,
    frame_width: int,
    frame_height: int,
) -> tuple[np.ndarray, str]:
    best_polygon: np.ndarray | None = None
    best_score = float("-inf")
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for _ in range(max(1, seed_frames)):
        ok, frame = capture.read()
        if not ok:
            break
        court_result = court_model.predict(
            source=frame,
            conf=court_conf,
            device=device,
            verbose=False,
        )[0]
        polygon = build_court_polygon(court_result)
        if polygon is None:
            continue
        _, confs, _ = result_boxes_to_numpy(court_result)
        score = float(np.sum(confs)) if len(confs) else 0.0
        if score > best_score:
            best_score = score
            best_polygon = polygon

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if best_polygon is not None:
        return best_polygon, "model"
    return default_court_polygon(frame_width, frame_height), "default"


def match_pose_indices_to_player_groups(
    players_detection_result,
    player_indices: list[int],
    pose_result,
    pose_indices: list[int],
) -> dict[int, int]:
    player_boxes, _, _ = result_boxes_to_numpy(players_detection_result)
    pose_boxes, _, _ = result_boxes_to_numpy(pose_result)
    pose_to_player: dict[int, int] = {}

    for pose_index in pose_indices:
        pose_box = pose_boxes[pose_index]
        best_player_index = -1
        best_score = float("-inf")
        for player_index in player_indices:
            player_box = player_boxes[player_index]
            if not pose_box_matches_player_box(pose_box, player_box):
                continue
            score = box_iou(pose_box, player_box)
            if score > best_score:
                best_score = score
                best_player_index = player_index
        if best_player_index >= 0:
            pose_to_player[pose_index] = best_player_index

    return pose_to_player


def make_frame_name(video_path: Path, frame_index: int) -> str:
    return f"{video_path.stem}_{frame_index:06d}.jpg"


def maybe_write_cvat_frame(frames_dir: Path | None, frame_name: str, frame: np.ndarray) -> None:
    if frames_dir is None:
        return
    frames_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frames_dir / frame_name), frame)


def collect_frame_annotations(
    *,
    frame_name: str,
    frame_width: int,
    frame_height: int,
    players_detection_result,
    player_det_indices: list[int],
    players_pose_result,
    player_pose_indices: list[int],
    ball_boxes: list[np.ndarray],
    court_result,
    group_id_start: int,
) -> tuple[FrameAnnotations, int]:
    frame_annotations = FrameAnnotations(name=frame_name, width=frame_width, height=frame_height)
    next_group_id = group_id_start

    player_boxes, _, _ = result_boxes_to_numpy(players_detection_result)
    pose_to_player = match_pose_indices_to_player_groups(
        players_detection_result,
        player_det_indices,
        players_pose_result,
        player_pose_indices,
    )
    player_group_ids: dict[int, int] = {}

    for player_index in player_det_indices:
        player_group_ids[player_index] = next_group_id
        next_group_id += 1
        box = player_boxes[player_index]
        frame_annotations.boxes.append(
            BoxAnnotation(
                label="player",
                xtl=float(box[0]),
                ytl=float(box[1]),
                xbr=float(box[2]),
                ybr=float(box[3]),
                group_id=player_group_ids[player_index],
            )
        )

    if players_pose_result.keypoints is not None:
        pose_keypoints = result_keypoints_to_numpy(players_pose_result)
        for pose_index in player_pose_indices:
            group_id = player_group_ids.get(pose_to_player.get(pose_index, -1))
            if group_id is None:
                group_id = next_group_id
                next_group_id += 1
            for keypoint_name, point in zip(COCO_KEYPOINT_NAMES, pose_keypoints[pose_index]):
                if np.any(point <= 0):
                    continue
                frame_annotations.points.append(
                    PointAnnotation(
                        label=keypoint_name,
                        x=float(point[0]),
                        y=float(point[1]),
                        group_id=group_id,
                    )
                )

    for box in ball_boxes:
        frame_annotations.boxes.append(
            BoxAnnotation(
                label="ball",
                xtl=float(box[0]),
                ytl=float(box[1]),
                xbr=float(box[2]),
                ybr=float(box[3]),
            )
        )

    for label, point in extract_court_points_by_name(court_result).items():
        frame_annotations.points.append(
            PointAnnotation(
                label=label,
                x=point[0],
                y=point[1],
            )
        )

    return frame_annotations, next_group_id


def indent_xml(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = indent + "  "
        for child in element:
            indent_xml(child, level + 1)
        if child is not None and (not child.tail or not child.tail.strip()):
            child.tail = indent
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = indent


def add_label(parent: ET.Element, name: str) -> None:
    label_node = ET.SubElement(parent, "label")
    ET.SubElement(label_node, "name").text = name


def write_cvat_xml(
    xml_path: Path,
    *,
    video_path: Path,
    frame_annotations: list[FrameAnnotations],
) -> None:
    xml_path.parent.mkdir(parents=True, exist_ok=True)

    annotations = ET.Element("annotations")
    ET.SubElement(annotations, "version").text = "1.1"

    meta = ET.SubElement(annotations, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = video_path.stem
    ET.SubElement(task, "mode").text = "annotation"
    ET.SubElement(task, "source").text = str(video_path)
    labels = ET.SubElement(task, "labels")

    label_names = ["player", "ball", *COCO_KEYPOINT_NAMES, *COURT_KEYPOINT_LABELS]
    for label_name in label_names:
        add_label(labels, label_name)

    ET.SubElement(meta, "dumped").text = "auto"

    for image_id, frame_annotation in enumerate(frame_annotations):
        image_node = ET.SubElement(
            annotations,
            "image",
            {
                "id": str(image_id),
                "name": frame_annotation.name,
                "width": str(frame_annotation.width),
                "height": str(frame_annotation.height),
            },
        )

        for box in frame_annotation.boxes:
            attrs = {
                "label": box.label,
                "source": "auto",
                "occluded": "0",
                "xtl": f"{box.xtl:.2f}",
                "ytl": f"{box.ytl:.2f}",
                "xbr": f"{box.xbr:.2f}",
                "ybr": f"{box.ybr:.2f}",
                "z_order": "0",
            }
            if box.group_id is not None:
                attrs["group_id"] = str(box.group_id)
            ET.SubElement(image_node, "box", attrs)

        for point in frame_annotation.points:
            attrs = {
                "label": point.label,
                "source": "auto",
                "occluded": "0",
                "points": f"{point.x:.2f},{point.y:.2f}",
                "z_order": "1",
            }
            if point.group_id is not None:
                attrs["group_id"] = str(point.group_id)
            ET.SubElement(image_node, "points", attrs)

    indent_xml(annotations)
    ET.ElementTree(annotations).write(xml_path, encoding="utf-8", xml_declaration=True)


def run_video(
    args: argparse.Namespace,
    *,
    video_path: Path,
    output_path: Path,
    xml_path: Path | None,
    cvat_frames_dir: Path | None,
    players_detection_model: YOLO,
    players_pose_model: YOLO,
    ball_model: YOLO,
    court_model: YOLO,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if xml_path is not None:
        xml_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    seeded_polygon, seed_mode = choose_seed_court_polygon(
        capture,
        court_model,
        court_conf=args.court_conf,
        device=args.device,
        seed_frames=args.court_seed_frames,
        frame_width=width,
        frame_height=height,
    )
    print(f"Court seed for {video_path.name}: {seed_mode}")

    player_det_memories: list[TrackMemory] = []
    player_pose_memories: list[TrackMemory] = []
    ball_memories: list[TrackMemory] = []
    ball_state = BallTrackState(center_history=deque(), size_history=deque())
    frame_annotations: list[FrameAnnotations] = []
    next_group_id = 1

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        if args.max_frames and frame_index > args.max_frames:
            break

        players_detection_result = players_detection_model.predict(
            source=frame,
            conf=args.players_conf,
            device=args.device,
            verbose=False,
        )[0]
        players_pose_result = players_pose_model.predict(
            source=frame,
            conf=args.pose_conf,
            device=args.device,
            verbose=False,
        )[0]
        ball_result = ball_model.predict(
            source=frame,
            conf=args.ball_conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        court_result = court_model.predict(
            source=frame,
            conf=args.court_conf,
            device=args.device,
            verbose=False,
        )[0]

        overlay = frame.copy()
        court_count = draw_court_keypoints(overlay, court_result)
        current_court_polygon = build_court_polygon(court_result)
        court_polygon = current_court_polygon if current_court_polygon is not None else seeded_polygon

        player_det_indices, player_det_memories = select_stable_indices(
            players_detection_result,
            court_polygon,
            player_det_memories,
            max_count=args.max_players,
            match_distance_ratio=0.12,
            min_match_iou=0.20,
            new_track_conf=max(args.players_conf + 0.15, 0.40),
            prefer_existing_bonus=0.60,
            frame_width=width,
            frame_height=height,
            point_selector=box_feet,
        )
        player_pose_indices, player_pose_memories = select_stable_indices(
            players_pose_result,
            court_polygon,
            player_pose_memories,
            max_count=args.max_players,
            match_distance_ratio=0.12,
            min_match_iou=0.15,
            new_track_conf=max(args.pose_conf + 0.15, 0.40),
            prefer_existing_bonus=0.60,
            frame_width=width,
            frame_height=height,
            point_selector=box_center,
        )
        player_pose_indices, player_pose_memories = filter_pose_indices_by_player_boxes(
            players_pose_result,
            player_pose_indices,
            player_pose_memories,
            players_detection_result,
            player_det_indices,
        )
        ball_indices, ball_memories = select_stable_indices(
            ball_result,
            court_polygon,
            ball_memories,
            max_count=max(1, args.max_balls) if args.max_balls != 0 else 0,
            match_distance_ratio=0.05,
            min_match_iou=0.00,
            new_track_conf=max(args.ball_conf + 0.20, 0.30),
            prefer_existing_bonus=0.90,
            frame_width=width,
            frame_height=height,
            point_selector=box_center,
        )
        ball_boxes, ball_confs, ball_state = select_ball_boxes(
            ball_result,
            ball_indices,
            court_polygon,
            ball_state,
            frame_width=width,
            frame_height=height,
            max_step_ratio=args.ball_max_step_ratio,
            smoothing_window=args.ball_smoothing_window,
            reacquire_gap_frames=args.ball_reacquire_gap_frames,
            max_reject_streak=args.ball_max_reject_streak,
        )

        players_count = draw_players_detection(overlay, players_detection_result, player_det_indices)
        pose_count = draw_players_pose(overlay, players_pose_result, player_pose_indices)
        ball_count = draw_ball_detection(overlay, ball_boxes, ball_confs)

        if current_court_polygon is None:
            fallback_polygon = seeded_polygon.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [fallback_polygon], isClosed=True, color=(180, 180, 180), thickness=2)
            draw_label(overlay, f"court fallback={seed_mode}", (20, 55), TEXT_COLOR)

        draw_label(
            overlay,
            (
                f"frame {frame_index}/{total_frames} | players_det={players_count} "
                f"players_pose={pose_count} ball={ball_count} court_pts={court_count}"
            ),
            (20, 30),
            TEXT_COLOR,
        )

        writer.write(overlay)

        if xml_path is not None:
            frame_name = make_frame_name(video_path, frame_index)
            frame_annotation, next_group_id = collect_frame_annotations(
                frame_name=frame_name,
                frame_width=width,
                frame_height=height,
                players_detection_result=players_detection_result,
                player_det_indices=player_det_indices,
                players_pose_result=players_pose_result,
                player_pose_indices=player_pose_indices,
                ball_boxes=ball_boxes,
                court_result=court_result,
                group_id_start=next_group_id,
            )
            frame_annotations.append(frame_annotation)
            maybe_write_cvat_frame(cvat_frames_dir, frame_name, frame)

        if frame_index % 25 == 0:
            print(f"Processed {video_path.name}: {frame_index}/{total_frames} frames")

    capture.release()
    writer.release()
    print(f"Overlay video written to: {output_path}")

    if xml_path is not None:
        write_cvat_xml(
            xml_path,
            video_path=video_path,
            frame_annotations=frame_annotations,
        )
        print(f"CVAT XML written to: {xml_path}")
        if cvat_frames_dir is not None:
            print(f"CVAT frame images written to: {cvat_frames_dir}")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    players_detection_path = require_file(args.players_detection_model, "Players detection model")
    players_pose_path = require_file(args.players_pose_model, "Players pose model")
    ball_path = require_file(args.ball_model, "Ball model")
    court_path = require_file(args.court_model, "Court model")

    jobs = collect_video_jobs(args)
    print(f"Running {len(jobs)} video job(s)")

    players_detection_model = YOLO(str(players_detection_path))
    players_pose_model = YOLO(str(players_pose_path))
    ball_model = YOLO(str(ball_path))
    court_model = YOLO(str(court_path))

    for job in jobs:
        run_video(
            args,
            video_path=job.video_path,
            output_path=job.output_path,
            xml_path=job.xml_path,
            cvat_frames_dir=job.cvat_frames_dir,
            players_detection_model=players_detection_model,
            players_pose_model=players_pose_model,
            ball_model=ball_model,
            court_model=court_model,
        )


if __name__ == "__main__":
    main()
