Test inputs  —  scripts/make_test_inputs.py
===========================================

FILE PURPOSES AND CONSISTENCY
------------------------------

portrait.jpg
  Used by:  I2V models  (--image)
  Role:     STARTING FRAME that the model extends into a video.
  Content:  Real photo (girl.png) — any interesting image works here.
  Pairs with: nothing specific; just needs to be a real scene.

character.jpg
  Used by:  WanAnimate  (--image)
  Role:     CHARACTER APPEARANCE — the person to animate.
  Content:  Full-body STANDING figure drawn at animation phase=0.
  Must match: pose_video.mp4 — same body geometry and standing pose.
  NOTE: I2V portrait (girl at café) must NOT be used here; she is seated
        and cannot perform the standing skeleton poses in pose_video.

pose_video.mp4
  Used by:  WanAnimate  (--pose-video)
  Role:     MOTION DRIVER — skeleton of the driving person's movements.
  Content:  DWPose COCO-17 skeleton, standing sway, 77 frames.
  Must match: face_video.mp4 — SAME animation phases, frame-for-frame.
              character.jpg — standing body geometry matches.

face_video.mp4
  Used by:  WanAnimate  (--face-video)
  Role:     FACE IDENTITY of the driver — whose expressions to project.
  Content:  Face crop of the same drawn figure at same animation phases.
  Must match: pose_video.mp4 — SAME phases, so face moves with the body.
  NOTE: Uses the drawn character's face, not the real girl from portrait.jpg.

video_ref.mp4
  Used by:  WanVACE  (--reference-video)
  Role:     SOURCE SCENE to edit.
  Content:  17 frames sampled from real motorcycle footage.
  Pairs with: video_mask.mp4 — same resolution (480×272) and frame count (17).

video_mask.mp4
  Used by:  WanVACE  (--mask-video)
  Role:     EDIT REGION — white=regenerate, black=keep.
  Content:  Oval sweeping left→right defines a moving inpaint region.
  Pairs with: video_ref.mp4 — same resolution and frame count.

KNOWN LIMITATION
----------------
For high-quality WanAnimate output you need a REAL driving video of a standing
person, DWPose-extracted pose_video, and a real face track.  No suitable real
driving video is available on this machine.  The three WanAnimate files are
internally consistent (all from the same drawn figure / same phases) but are
synthetic — suitable for pipeline smoke testing only.

EXAMPLE COMMANDS  (--num-inference-steps 5 = smoke test)
---------------------------------------------------------

T2V:
  python scripts/infer.py --model wan1.3b     --method dense      --num-inference-steps 5
  python scripts/infer.py --model wan14b      --method svoo       --num-inference-steps 5
  python scripts/infer.py --model wan22       --method svg2       --num-inference-steps 5
  python scripts/infer.py --model hunyuan     --method adacluster --num-inference-steps 5
  python scripts/infer.py --model skyreels-v2 --method radial     --num-inference-steps 5
  python scripts/infer.py --model cogvideox   --method dense      --num-inference-steps 5
  python scripts/infer.py --model ltx         --method dense      --num-inference-steps 5
  python scripts/infer.py --model allegro     --method dense      --num-inference-steps 5
  python scripts/infer.py --model mochi       --method dense      --num-inference-steps 5
  python scripts/infer.py --model easyanimate --method dense      --num-inference-steps 5

I2V  (portrait.jpg = real starting frame):
  python scripts/infer.py --model wan14b-i2v    --method svoo  --image test_inputs/portrait.jpg --num-inference-steps 5
  python scripts/infer.py --model wan22-i2v     --method dense --image test_inputs/portrait.jpg --num-inference-steps 5
  python scripts/infer.py --model hunyuan-i2v   --method dense --image test_inputs/portrait.jpg --num-inference-steps 5
  python scripts/infer.py --model skyreels-i2v  --method dense --image test_inputs/portrait.jpg --num-inference-steps 5
  python scripts/infer.py --model cogvideox-i2v --method dense --image test_inputs/portrait.jpg --num-inference-steps 5
  python scripts/infer.py --model ltx-i2v       --method dense --image test_inputs/portrait.jpg --num-inference-steps 5

WanVACE  (video_ref + video_mask are paired):
  python scripts/infer.py --model wan-vace --method dense \
      --reference-video test_inputs/video_ref.mp4 \
      --mask-video test_inputs/video_mask.mp4 --num-inference-steps 5

WanAnimate  (character.jpg + pose_video + face_video are all aligned):
  python scripts/infer.py --model wananimate --method dense \
      --image test_inputs/character.jpg \
      --pose-video test_inputs/pose_video.mp4 \
      --face-video test_inputs/face_video.mp4 \
      --num-inference-steps 5
