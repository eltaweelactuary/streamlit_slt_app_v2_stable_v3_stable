/**
 * Three.js Bridge for ReadyPlayerMe VRM Rigging
 * 🤖🕶️✨ Cinema Mode 3D Renderer
 * 
 * This script creates a WebGL scene, loads a VRM model,
 * and animates it using skeletal DNA data passed from the Streamlit backend.
 */

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { GLTFLoader } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from 'https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@2.0.6/lib/three-vrm.module.js';

// --- Configuration ---
const AVATAR_URL = 'https://models.readyplayer.me/6478f5f7a23d4cbdef3e45c1.vrm'; // Default RPM avatar
const CANVAS_ID = 'threejs-canvas';

// --- Three.js Setup ---
let scene, camera, renderer, vrm, clock;
let dnaFrames = [];
let currentFrameIndex = 0;
let isPlaying = false;

function init() {
    clock = new THREE.Clock();
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e); // Dark immersive background

    // Camera
    camera = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 1.0, 3);
    camera.lookAt(0, 1, 0);

    // Renderer
    const canvas = document.getElementById(CANVAS_ID);
    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);
    directionalLight.position.set(1, 2, 1).normalize();
    scene.add(directionalLight);

    // Load VRM Model
    loadAvatar(AVATAR_URL);

    // Animate Loop
    animate();
}

function loadAvatar(url) {
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));

    loader.load(
        url,
        (gltf) => {
            vrm = gltf.userData.vrm;
            VRMUtils.rotateVRM0(vrm); // Rotate if VRM 0.x
            scene.add(vrm.scene);
            console.log('🤖 ReadyPlayerMe Avatar Loaded!', vrm);
        },
        (progress) => console.log('Loading avatar...', (progress.loaded / progress.total) * 100 + '%'),
        (error) => console.error('Error loading avatar:', error)
    );
}

function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();

    if (vrm) {
        vrm.update(delta);

        if (isPlaying && dnaFrames.length > 0) {
            applyDnaFrame(dnaFrames[currentFrameIndex]);
            currentFrameIndex = (currentFrameIndex + 1) % dnaFrames.length;
        }
    }

    renderer.render(scene, camera);
}

/**
 * Maps skeletal DNA data to VRM bone rotations.
 * This is the core rigging function.
 * @param {object} frame - A single frame of DNA data { left_hand, right_hand, pose }
 */
function applyDnaFrame(frame) {
    if (!vrm || !frame) return;

    const humanoid = vrm.humanoid;

    // --- Pose (Body) Rigging ---
    // MediaPipe Pose indices: 11=LeftShoulder, 12=RightShoulder, 13=LeftElbow, etc.
    // This is a simplified mapping. A full implementation would require inverse kinematics.

    const pose = frame.pose;
    if (pose && pose.length >= 99) { // 33 points * 3 coords
        // Upper Arm Rotation (Simplified: use shoulder-to-elbow vector)
        const leftShoulder = new THREE.Vector3(pose[11 * 3], -pose[11 * 3 + 1], pose[11 * 3 + 2]);
        const leftElbow = new THREE.Vector3(pose[13 * 3], -pose[13 * 3 + 1], pose[13 * 3 + 2]);

        const leftArmDir = new THREE.Vector3().subVectors(leftElbow, leftShoulder).normalize();
        const leftArmBone = humanoid.getNormalizedBoneNode('leftUpperArm');
        if (leftArmBone) {
            // Apply rotation - this is a placeholder for full IK
            leftArmBone.rotation.z = -Math.acos(leftArmDir.y) * 0.5;
        }

        const rightShoulder = new THREE.Vector3(pose[12 * 3], -pose[12 * 3 + 1], pose[12 * 3 + 2]);
        const rightElbow = new THREE.Vector3(pose[14 * 3], -pose[14 * 3 + 1], pose[14 * 3 + 2]);

        const rightArmDir = new THREE.Vector3().subVectors(rightElbow, rightShoulder).normalize();
        const rightArmBone = humanoid.getNormalizedBoneNode('rightUpperArm');
        if (rightArmBone) {
            rightArmBone.rotation.z = Math.acos(rightArmDir.y) * 0.5;
        }
    }

    // --- Hand Rigging (Future Enhancement) ---
    // frame.left_hand and frame.right_hand contain 21 points each.
    // A full implementation would map these to finger bone rotations.
}

/**
 * Receives DNA data from the Streamlit backend and starts playback.
 * @param {Array} dna - The full DNA sequence (array of frame objects).
 */
window.receiveAndPlayDna = function (dna) {
    console.log('🧬 DNA Received!', dna.length, 'frames.');
    dnaFrames = dna;
    currentFrameIndex = 0;
    isPlaying = true;
};

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
