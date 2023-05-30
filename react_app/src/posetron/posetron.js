import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';
import React, { useRef, useState, useEffect } from 'react'
import Webcam from 'react-webcam'
import './posetron.css'
const POINTS = {
  NOSE: 0,
  LEFT_EYE: 1,
  RIGHT_EYE: 2,
  LEFT_EAR: 3,
  RIGHT_EAR: 4,
  LEFT_SHOULDER: 5,
  RIGHT_SHOULDER: 6,
  LEFT_ELBOW: 7,
  RIGHT_ELBOW: 8,
  LEFT_WRIST: 9,
  RIGHT_WRIST: 10,
  LEFT_HIP: 11,
  RIGHT_HIP: 12,
  LEFT_KNEE: 13,
  RIGHT_KNEE: 14,
  LEFT_ANKLE: 15,
  RIGHT_ANKLE: 16,
}

const keypointConnections = {
  nose: ['left_ear', 'right_ear'],
  left_ear: ['left_shoulder'],
  right_ear: ['right_shoulder'],
  left_shoulder: ['right_shoulder', 'left_elbow', 'left_hip'],
  right_shoulder: ['right_elbow', 'right_hip'],
  left_elbow: ['left_wrist'],
  right_elbow: ['right_wrist'],
  left_hip: ['left_knee', 'right_hip'],
  right_hip: ['right_knee'],
  left_knee: ['left_ankle'],
  right_knee: ['right_ankle']
}

function drawSegment(context, [mx, my], [tx, ty], color) {
  context.beginPath()
  context.moveTo(mx, my)
  context.lineTo(tx, ty)
  context.lineWidth = 5
  context.strokeStyle = color
  context.stroke()
}
function drawPoint(context, x, y, r, color) {
  context.beginPath();
  context.arc(x, y, r, 0, 2 * Math.PI);
  context.fillStyle = color;
  context.fill();
}


function DropDown({ poses, currentPosture, setcurrentPosture }) {
  return (
    <div
      className='dropdown dropdown-container'>
      <button
        className="btn btn-secondary dropdown-toggle"
        type='button'
        data-bs-toggle="dropdown"
        id="pose-dropdown-btn"
        aria-expanded="false"
      >{currentPosture}
      </button>
      <ul className="dropdown-menu dropdown-custom-menu" aria-labelledby="dropdownMenuButton1">
        {poses.map((pose) => (
          <li onClick={() => setcurrentPosture(pose)}>
            <div className="dropdown-item-container">
              <p className="dropdown-item-1">{pose}</p>
            </div>
          </li>
        ))}

      </ul>


    </div>
  )
}

let skeleton = 'rgb(255,255,255)'
let poses = [
  'hip', 'shoulder'
]

let interval

let flag = false


function Posetron() {
  const webcam = useRef(null)
  const canvas = useRef(null)


  const [startTime, setstartTime] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [bestMatch, setbestMatch] = useState(0)
  const [currentPosture, setcurrentPosture] = useState('hip')
  const [PoseStart, detectStart] = useState(false)


  useEffect(() => {
    const timeDifference = (currentTime - startTime) / 1000
    if (flag) {
    }
    if ((currentTime - startTime) / 1000 > bestMatch) {
      setbestMatch(timeDifference)
    }
    // eslint-disable-next-line
  }, [currentTime])


  useEffect(() => {
    setCurrentTime(0)
    setbestMatch(0)
  }, [currentPosture])

  const CLASS_NO = {
    hip:0,
    shoulder:1,
  }

  function get_center_point(landmarks, left_bodypart, right_bodypart) {
    let left = tf.gather(landmarks, left_bodypart, 1)
    let right = tf.gather(landmarks, right_bodypart, 1)
    const center = tf.add(tf.mul(left, 0.5), tf.mul(right, 0.5))
    return center

  }

  function get_pose_size(landmarks, torso_size_multiplier = 2.5) {
    let hips_center = get_center_point(landmarks, POINTS.LEFT_HIP, POINTS.RIGHT_HIP)
    let shoulders_center = get_center_point(landmarks, POINTS.LEFT_SHOULDER, POINTS.RIGHT_SHOULDER)
    let torso_size = tf.norm(tf.sub(shoulders_center, hips_center))
    let pose_center_new = get_center_point(landmarks, POINTS.LEFT_HIP, POINTS.RIGHT_HIP)
    pose_center_new = tf.expandDims(pose_center_new, 1)

    pose_center_new = tf.broadcastTo(pose_center_new,
      [1, 17, 2]
    )
    let dist = tf.gather(tf.sub(landmarks, pose_center_new), 0, 0)
    let max_dist = tf.max(tf.norm(dist, 'euclidean', 0))

    let pose_size = tf.maximum(tf.mul(torso_size, torso_size_multiplier), max_dist)
    return pose_size
  }

  function normalize_pose_landmarks(landmarks) {
    let pose_center = get_center_point(landmarks, POINTS.LEFT_HIP, POINTS.RIGHT_HIP)
    pose_center = tf.expandDims(pose_center, 1)
    pose_center = tf.broadcastTo(pose_center,
      [1, 17, 2]
    )
    landmarks = tf.sub(landmarks, pose_center)

    let pose_size = get_pose_size(landmarks)
    landmarks = tf.div(landmarks, pose_size)
    return landmarks
  }

  function landmarks_to_embedding(landmarks) {
    landmarks = normalize_pose_landmarks(tf.expandDims(landmarks, 0))
    let embedding = tf.reshape(landmarks, [1, 34])
    return embedding
  }

  const runMovenet = async () => {
    const detectorConfig = { modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER };
    const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
    const poseclassifier = await tf.loadLayersModel('https://posetronmodelbucket.s3.ap-south-1.amazonaws.com/model/model.json')
    interval = setInterval(() => {
      detectPosture(detector, poseclassifier)
    }, 100)
  }

  const detectPosture = async (detector, poseclassifier) => {
    if (
      typeof webcam.current !== "undefined" &&
      webcam.current !== null &&
      webcam.current.video.readyState === 4
    ) {
      let failcase = 0
      const video = webcam.current.video
      const pose = await detector.estimatePoses(video)
      const context = canvas.current.getContext('2d')
      context.clearRect(0, 0, canvas.current.width, canvas.current.height);
      try {
        const keypoints = pose[0].keypoints
        let input = keypoints.map((keypoint) => {
          if (keypoint.score > 0.4) {
            if (!(keypoint.name === 'left_eye' || keypoint.name === 'right_eye')) {
              drawPoint(context, keypoint.x, keypoint.y, 8, 'rgb(255,255,255)')
              let connections = keypointConnections[keypoint.name]
              try {
                connections.forEach((connection) => {
                  let conName = connection.toUpperCase()
                  drawSegment(context, [keypoint.x, keypoint.y],
                    [keypoints[POINTS[conName]].x,
                    keypoints[POINTS[conName]].y]
                    , skeleton)
                })
              } catch (err) {

              }

            }
          } else {
            failcase += 1
          }
          return [keypoint.x, keypoint.y]
        })
        if (failcase > 4) {
          skeleton = 'rgb(255,255,255)'
          return
        }
        const processedInput = landmarks_to_embedding(input)
        const classification = poseclassifier.predict(processedInput)

        classification.array().then((data) => {
          const classNo = CLASS_NO[currentPosture]
          console.log(data[0][classNo])
          if (data[0][classNo] > 0.1) {

            if (!flag) {
              setstartTime(new Date(Date()).getTime())
              flag = true
            }
            setCurrentTime(new Date(Date()).getTime())
            skeleton = 'rgb(255,0,0)'
          } else {
            flag = false
            skeleton = 'rgb(255,255,255)'
          }
        })
      } catch (err) {
        console.log(err)
      }


    }
  }

  function startYoga() {
    detectStart(true)
    runMovenet()
  }

  function stopPose() {
    detectStart(false)
    clearInterval(interval)
  }



  if (PoseStart) {
    return (
      <div className="yoga-container">

        <div>

          <Webcam
            width='640px'
            height='480px'
            id="webcam"
            ref={webcam}
            style={{
              position: 'absolute',
              left: 120,
              top: 100,
              padding: '0px',
            }}
          />
          <canvas
            ref={canvas}
            id="my-canvas"
            width='640px'
            height='480px'
            style={{
              position: 'absolute',
              left: 120,
              top: 100,
              zIndex: 1
            }}
          >
          </canvas>

        </div>
        <button
          onClick={stopPose}
          className="secondary-btn"
        >Stop Detection</button>
      </div>
    )
  }

  return (
    <div
      className="yoga-container"
    >
      <DropDown
        poses={poses}
        currentPosture={currentPosture}
        setcurrentPosture={setcurrentPosture}
      />
      <button
        onClick={startYoga}
        className="secondary-btn"
      >Start Detection</button>
    </div>
  )
}

export default Posetron