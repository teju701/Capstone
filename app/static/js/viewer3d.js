/**
 * viewer3d.js
 * ───────────
 * Real-Time Three.js WebGL 3D Point Cloud & Scene Visualizer.
 * Supports Orbit Controls, BEV camera view, and dynamic colormaps.
 */

class PointCloudViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.pointsMesh = null;
        this.rawPointData = null;
        this.currentMode = 'rgb'; // 'rgb' | 'seg' | 'depth'
        this.pointSize = 3.0;
        this.isBEV = false;

        this.init();
    }

    init() {
        const width = this.container.clientWidth || 600;
        const height = this.container.clientHeight || 300;

        // 1. Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x05070d);

        // Grid floor helper
        const grid = new THREE.GridHelper(60, 30, 0x06b6d4, 0x1e293b);
        grid.position.y = -2.5;
        this.scene.add(grid);

        // 2. Camera (Driver ego-perspective)
        this.camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 200);
        this.resetCamera();

        // 3. WebGL Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.container.appendChild(this.renderer.domElement);

        // 4. Orbit Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.maxDistance = 100;
        this.controls.minDistance = 1;

        // Window resize listener
        window.addEventListener('resize', () => this.onResize());

        // Animation Loop
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }

    resetCamera() {
        this.isBEV = false;
        this.camera.position.set(0, 1.2, 3.5);
        this.camera.lookAt(0, 0, -15);
        if (this.controls) this.controls.target.set(0, 0, -15);
    }

    setBEVView() {
        this.isBEV = true;
        this.camera.position.set(0, 35, -20);
        this.camera.lookAt(0, 0, -20);
        if (this.controls) this.controls.target.set(0, 0, -20);
    }

    onResize() {
        if (!this.container || !this.renderer || !this.camera) return;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    updateData(pointData) {
        this.rawPointData = pointData; // array of 8 elements per point: [X, Y, Z, r, g, b, class_id, dist_m]
        this.rebuildPointsMesh();
    }

    rebuildPointsMesh() {
        if (!this.rawPointData || this.rawPointData.length === 0) return;

        if (this.pointsMesh) {
            this.scene.remove(this.pointsMesh);
            this.pointsMesh.geometry.dispose();
            this.pointsMesh.material.dispose();
        }

        const numPoints = Math.floor(this.rawPointData.length / 8);
        const positions = new Float32Array(numPoints * 3);
        const colors = new Float32Array(numPoints * 3);

        const lblCount = document.getElementById('lbl-point-count');
        if (lblCount) lblCount.textContent = `Points: ${numPoints.toLocaleString()}`;

        for (let i = 0; i < numPoints; i++) {
            const idx = i * 8;
            const x = this.rawPointData[idx + 0];
            const y = this.rawPointData[idx + 1];
            const z = this.rawPointData[idx + 2];
            const r = this.rawPointData[idx + 3];
            const g = this.rawPointData[idx + 4];
            const b = this.rawPointData[idx + 5];
            const classId = this.rawPointData[idx + 6];
            const distMeters = this.rawPointData[idx + 7];

            positions[i * 3 + 0] = x;
            positions[i * 3 + 1] = y;
            positions[i * 3 + 2] = z;

            let cr = r, cg = g, cb = b;

            if (this.currentMode === 'seg') {
                const cMeta = getClassInfo(classId);
                cr = cMeta.color[0] / 255.0;
                cg = cMeta.color[1] / 255.0;
                cb = cMeta.color[2] / 255.0;
            } else if (this.currentMode === 'depth') {
                // Elevation / distance ramp (red near -> cyan mid -> blue far)
                const normD = Math.min(Math.max((distMeters - 1.0) / 45.0, 0.0), 1.0);
                cr = Math.sin(normD * Math.PI);
                cg = Math.sin(normD * Math.PI * 0.7);
                cb = 1.0 - normD;
            }

            colors[i * 3 + 0] = cr;
            colors[i * 3 + 1] = cg;
            colors[i * 3 + 2] = cb;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: this.pointSize * 0.05,
            vertexColors: true,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.95
        });

        this.pointsMesh = new THREE.Points(geometry, material);
        this.scene.add(this.pointsMesh);
    }

    setColorMode(mode) {
        this.currentMode = mode;
        this.rebuildPointsMesh();
    }

    setPointSize(size) {
        this.pointSize = size;
        if (this.pointsMesh && this.pointsMesh.material) {
            this.pointsMesh.material.size = size * 0.05;
        }
    }

    animate() {
        requestAnimationFrame(this.animate);
        if (this.controls) this.controls.update();
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }
}
