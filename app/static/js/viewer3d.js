/**
 * viewer3d.js
 * ───────────
 * Real-Time Three.js WebGL 3D Point Cloud & Autonomous Driving Scene Visualizer.
 * Supports Driver Cockpit POV, 3rd-Person Follow Cam, Bird's Eye View (BEV),
 * and dynamic RGB/Semantic/Depth color modes.
 */

class PointCloudViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.pointsMesh = null;
        this.egoVehicle = null;
        this.rawPointData = null;
        this.currentMode = 'rgb'; // 'rgb' | 'seg' | 'depth'
        this.pointSize = 3.5;
        this.currentView = 'follow'; // 'follow' | 'cockpit' | 'bev'

        this.init();
    }

    init() {
        const width = this.container.clientWidth || 600;
        const height = this.container.clientHeight || 320;

        // 1. Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x05070d);
        this.scene.fog = new THREE.FogExp2(0x05070d, 0.015);

        // Ground Plane Grid (Y = 0.0m matches road surface)
        const grid = new THREE.GridHelper(80, 40, 0x00f0ff, 0x1e293b);
        grid.position.y = 0.0;
        this.scene.add(grid);

        // Ego-Vehicle representation (Wireframe bounding box representing autonomous vehicle)
        const egoGroup = new THREE.Group();
        const carGeo = new THREE.BoxGeometry(1.8, 1.4, 4.2);
        const carMat = new THREE.MeshBasicMaterial({
            color: 0x00f0ff,
            wireframe: true,
            transparent: true,
            opacity: 0.55
        });
        const carMesh = new THREE.Mesh(carGeo, carMat);
        carMesh.position.set(0, 0.7, 1.2);
        egoGroup.add(carMesh);

        // Headlight direction cones / rays
        const rayMat = new THREE.LineBasicMaterial({ color: 0x00f0ff, transparent: true, opacity: 0.35 });
        const rayGeo1 = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(-0.7, 0.6, -0.9),
            new THREE.Vector3(-1.8, 0.1, -10.0)
        ]);
        const rayGeo2 = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0.7, 0.6, -0.9),
            new THREE.Vector3(1.8, 0.1, -10.0)
        ]);
        egoGroup.add(new THREE.Line(rayGeo1, rayMat));
        egoGroup.add(new THREE.Line(rayGeo2, rayMat));

        this.egoVehicle = egoGroup;
        this.scene.add(this.egoVehicle);

        // 2. Perspective Camera
        this.camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 200);
        this.setFollowView();

        // 3. WebGL Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.container.appendChild(this.renderer.domElement);

        // 4. Orbit Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.maxDistance = 90;
        this.controls.minDistance = 0.5;
        this.controls.target.set(0, 0.5, -14);

        // Window resize listener
        window.addEventListener('resize', () => this.onResize());

        // Animation Loop
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }

    setFollowView() {
        this.currentView = 'follow';
        this.camera.position.set(0, 3.2, 6.5);
        if (this.controls) {
            this.controls.target.set(0, 0.5, -14);
            this.controls.update();
        }
    }

    setCockpitView() {
        this.currentView = 'cockpit';
        this.camera.position.set(0, 1.2, 0.2);
        if (this.controls) {
            this.controls.target.set(0, 1.0, -18);
            this.controls.update();
        }
    }

    setBEVView() {
        this.currentView = 'bev';
        this.camera.position.set(0, 32, -15);
        if (this.controls) {
            this.controls.target.set(0, 0, -15.01);
            this.controls.update();
        }
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
                // Distance color ramp (Red <5m -> Yellow 5-15m -> Green 15-30m -> Cyan/Blue >30m)
                const normD = Math.min(Math.max((distMeters - 1.0) / 45.0, 0.0), 1.0);
                cr = Math.max(0.0, 1.0 - normD * 2.0);
                cg = Math.min(normD * 2.0, 2.0 - normD * 2.0);
                cb = Math.max(0.0, (normD - 0.5) * 2.0);
            }

            colors[i * 3 + 0] = cr;
            colors[i * 3 + 1] = cg;
            colors[i * 3 + 2] = cb;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: this.pointSize * 0.045,
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
            this.pointsMesh.material.size = size * 0.045;
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
