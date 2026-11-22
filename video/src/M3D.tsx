// "One camera -> every car in metres": Ultralytics YOLO26x-seg + YOLO26x-depth (TensorRT FP16) on KITTI raw drive
// 0009 (a pre-registered test drive the depth model never saw). The drive plays in real time (frames 230-425); while
// it drives, the frame lifts into metric 3D, the depth model finds the road (camera height), every car gets a ground
// position, the laser scanner and the human 3D labels check it, and crossing cars get speeds. Every frame's 3D sits in
// one world through the car's pose, so parked cars stay put as we drive past. All positions/speeds are the exact
// test-run values (export.py).
import React, {CSSProperties, useEffect, useMemo, useState} from 'react';
import {AbsoluteFill, Img, continueRender, delayRender, Easing, interpolate, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {ThreeCanvas} from '@remotion/three';
import {useThree} from '@react-three/fiber';
import * as THREE from 'three';
import {loadFont as loadOnest} from '@remotion/google-fonts/Onest';
import {loadFont as loadMono} from '@remotion/google-fonts/JetBrainsMono';
import SCENE from '../public/scene.json';

const onest = loadOnest('normal', {weights: ['400', '600', '800'], subsets: ['latin']});
const mono = loadMono('normal', {weights: ['400', '600'], subsets: ['latin']});
const SANS = onest.fontFamily, MONO = mono.fontFamily;

export const FPS = 30;
// story beats in seconds of drive time (drive frame = 230 + 10 s); crossing traffic starts at frame 360 = TL.run
const TL = {lift: 2.6, road: 5.0, pins: 7.5, proof: 10.0, run: 13.0, res: 20.0, res2: 23.0, fail: 26.0, outro: 30.0, end: 34.0};
// the first 0.4 s is a cover frame (X shows frame 0): the laser check
const COVER = 0.4, COVER_AT = 13.9;
export const TOTAL = Math.round((TL.end + COVER) * FPS);
const BG = '#05070C', TEXT = '#F4F1EA', DIM = '#8A929C', CAR = '#FF8A4C', PERSON = '#2ECC71', ROAD = '#3CC6C9', NV = '#76B900';
// test numbers (TEST.md, run 2): centre error 0-80 m and speed error, all 11 test drives
const RES = {cars: 523, hy: '1.2', b0: '2.0', kmh: '1.7', kmhB0: '3.0', ms: '23.5', drives: 11};
const REPO = 'github.com/zalkarbek2023/metric3d';

type Obj = {det: number; track: number; cls: string; box: number[]; c: Record<'depth' | 'hybrid' | 'b0', number[]>;
  gt_tid: number | null; err?: Record<string, number>; kmh?: Record<string, number>; trail: number[][]};
type GT = {tid: number; type: string; corners: number[][]; centre: number[]};
type Frame = {k: number; frame: number; objects: Obj[]; gt: GT[]; plane: number[] | null; pose: number[]};
const D = SCENE as unknown as {size: number[]; K: number[][]; grid: number[]; grid_step: number; lidar_idx: number[][];
  frames: Frame[]; cam_h_true: number};
const [IW, IH] = D.size;
const FX = D.K[0][0], FY = D.K[1][1], CX = D.K[0][2], CY = D.K[1][2];
const NF = D.frames.length;

const CLAMP = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;
const OUT = Easing.bezier(0.16, 1, 0.3, 1);
const INOUT = Easing.bezier(0.65, 0, 0.35, 1);
const ramp = (s: number, a: number, b: number, e = OUT) => interpolate(s, [a, b], [0, 1], {...CLAMP, easing: e});
const win = (s: number, a: number, b: number, fade = 0.45) => ramp(s, a, a + fade) * (1 - ramp(s, b - fade, b, INOUT));
const pad = (k: number) => String(k).padStart(3, '0');

// drive time: real time, one drive frame per 0.1 s, held on the last frame
const kAt = (s: number) => Math.min(NF - 1, Math.max(0, s * 10));

// camera-2 (x right, y down, z forward) -> three (x right, y up, z back)
const v3 = (x: number, y: number, z: number) => new THREE.Vector3(x, -y, -z);
// height of the frame's road plane under ground point (x, z) of camera-2, as a three y
const groundY = (pl: number[] | null, x: number, z: number) => {
  if (!pl) return -D.cam_h_true;
  const [nx, ny, nz, d0] = pl;
  return (d0 + nx * x + nz * z) / ny;  // y_cam = -(d0 + nx x + nz z) / ny ; three y = -y_cam
};

// ---------------------------------------------------------------- the drive in one world
// world = three axes of camera-2 at the first exported frame; T[k] maps frame k's camera (three axes) into it
const FLIP = new THREE.Matrix4().makeScale(1, -1, -1);
const T = D.frames.map((fr) => {
  const p = fr.pose;
  const m = new THREE.Matrix4().set(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], 0, 0, 0, 1);
  return FLIP.clone().multiply(m).multiply(FLIP);
});
const TP = T.map((m) => {
  const p = new THREE.Vector3(), q = new THREE.Quaternion(), sc = new THREE.Vector3();
  m.decompose(p, q, sc);
  return {p, q};
});
const yawOf = (m: THREE.Matrix4) => {
  const f = new THREE.Vector3(0, 0, -1).transformDirection(m);
  return Math.atan2(-f.x, -f.z);
};
// camera height from each frame's road plane, median over +-0.5 s
const H_EST = D.frames.map((_, k) => {
  const v = D.frames.slice(Math.max(0, k - 5), k + 6).map((f) => f.plane?.[3]).filter((x): x is number => x != null).sort((a, b) => a - b);
  return v.length ? v[Math.floor(v.length / 2)] : D.cam_h_true;
});
type EgoPose = {m: THREE.Matrix4; pos: THREE.Vector3; yaw: number; h: number};
// our car between two drive frames: position and heading only, so the virtual camera does not shake with pitch/roll
const egoAt = (kf: number): EgoPose => {
  const k = Math.min(NF - 1, Math.floor(kf)), k1 = Math.min(NF - 1, k + 1), al = Math.min(1, kf - k);
  const pos = TP[k].p.clone().lerp(TP[k1].p, al);
  const yaw = yawOf(T[k]) + (yawOf(T[k1]) - yawOf(T[k])) * al;
  const h = H_EST[k] + (H_EST[k1] - H_EST[k]) * al;
  return {m: new THREE.Matrix4().makeRotationY(yaw).setPosition(pos), pos, yaw, h};
};
// a car's ground point (x, z of frame k's camera-2) in the world
const onGround = (k: number, x: number, z: number) => new THREE.Vector3(x, groundY(D.frames[k].plane, x, z), -z).applyMatrix4(T[k]);
// first and last drive frame of every track: pins fade in and out instead of popping
const SPAN = new Map<number, [number, number]>();
D.frames.forEach((fr, k) => fr.objects.forEach((o) => {
  const sp = SPAN.get(o.track);
  SPAN.set(o.track, sp ? [sp[0], k] : [k, k]);
}));

// ---------------------------------------------------------------- assets
type Assets = {depth: Uint16Array; lidar: Int16Array; photos: THREE.Texture[]};
// everything is loaded before the canvas mounts: ThreeCanvas does not redraw after an async texture load
function useAssets(): Assets | null {
  const [a, setA] = useState<Assets | null>(null);
  const [h] = useState(() => delayRender('assets'));
  useEffect(() => {
    const L = new THREE.TextureLoader();
    const photo = (k: number) => new Promise<THREE.Texture>((res) => L.load(staticFile(`img/${pad(k)}.jpg`), (t) => {
      t.colorSpace = THREE.SRGBColorSpace; res(t);
    }));
    Promise.all([
      fetch(staticFile('depth.bin')).then((r) => r.arrayBuffer()),
      fetch(staticFile('lidar.bin')).then((r) => r.arrayBuffer()),
      Promise.all(D.frames.map((_, k) => photo(k))),
      onest.waitUntilDone(), mono.waitUntilDone(),
    ]).then(([d, l, photos]) => { setA({depth: new Uint16Array(d), lidar: new Int16Array(l), photos}); continueRender(h); });
  }, [h]);
  return a;
}
const labelCache = new Map<string, THREE.CanvasTexture>();
const labelFrac = new Map<string, number>();  // width of the drawn plate / width of the sprite
const label = (text: string, color: string) => {
  const key = text + color;
  let t = labelCache.get(key);
  if (!t) {
    const c = document.createElement('canvas');
    c.width = 640; c.height = 128;
    const g = c.getContext('2d')!;
    g.font = `600 56px ${MONO}`;
    const w = Math.min(620, g.measureText(text).width + 64);
    labelFrac.set(key, w / 640);
    g.fillStyle = 'rgba(5,7,12,0.86)';
    g.beginPath(); g.roundRect((640 - w) / 2, 16, w, 96, 20); g.fill();
    g.fillStyle = color; g.fillRect((640 - w) / 2, 16, 12, 96);
    g.fillStyle = '#ffffff'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText(text, 320 + 6, 66);
    t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    labelCache.set(key, t);
  }
  return t;
};

// ---------------------------------------------------------------- camera path (in our car's frame: it rides along)
type Cam = {pos: THREE.Vector3; look: THREE.Vector3; hfov: number};
const orbitAt = (target: THREE.Vector3, deg: number, r: number, h: number, hfov = 62): Cam => {
  const a = (deg * Math.PI) / 180;
  return {pos: target.clone().add(new THREE.Vector3(r * Math.sin(a), h, r * Math.cos(a))), look: target.clone(), hfov};
};
const WIDE = (deg: number) => orbitAt(new THREE.Vector3(1.5, -1.7, -14), deg, 21, 8.5);
const NEAR: Cam = {pos: new THREE.Vector3(-6, 3.5, 6), look: new THREE.Vector3(1, -1.2, -8), hfov: 70};
const DASH: Cam = {pos: new THREE.Vector3(0, 0, 0), look: new THREE.Vector3(0, 0, -30), hfov: (2 * Math.atan(IW / 2 / FX) * 180) / Math.PI};
const chase = (s: number): Cam => ({
  pos: new THREE.Vector3(1.0 * Math.sin(s * 0.35), 12.5, 7.5), look: new THREE.Vector3(0.5, -1.7, -17), hfov: 64,
});
const mix = (a: Cam, b: Cam, t: number): Cam => ({pos: a.pos.clone().lerp(b.pos, t), look: a.look.clone().lerp(b.look, t), hfov: a.hfov + (b.hfov - a.hfov) * t});
function camAt(s: number): Cam {
  if (s < TL.lift) return DASH;
  if (s < TL.road) return mix(DASH, WIDE(-24 + 6 * ramp(s, TL.road - 0.6, TL.road)), ramp(s, TL.lift + 0.2, TL.road - 0.6, INOUT));
  if (s < TL.pins) return mix(WIDE(-18), NEAR, ramp(s, TL.road, TL.road + 0.8, INOUT) * (1 - ramp(s, TL.pins - 1.0, TL.pins, INOUT)));
  if (s < TL.proof) return mix(WIDE(-18), WIDE(-10), ramp(s, TL.pins, TL.proof, INOUT));
  if (s < TL.run) return WIDE(-10 + 34 * ramp(s, TL.proof, TL.run, INOUT));
  if (s < 19.2) return mix(WIDE(24), chase(s), ramp(s, TL.run, 19.2, INOUT));
  const c = chase(s);
  const back = ramp(s, TL.res, TL.outro, INOUT);
  c.pos.add(new THREE.Vector3(0, 4 * back, 5 * back));
  return c;
}
const toWorld = (c: Cam, ego: EgoPose): Cam => ({pos: c.pos.clone().applyMatrix4(ego.m), look: c.look.clone().applyMatrix4(ego.m), hfov: c.hfov});
const vfovOf = (hfov: number, aspect: number) => (2 * Math.atan(Math.tan((hfov * Math.PI) / 360) / aspect) * 180) / Math.PI;

// yoff > 0 moves the picture up by that fraction of the view (the bottom of the view sits under the text)
const shiftView = (c: THREE.PerspectiveCamera, aspect: number, yoff: number) => {
  if (yoff > 0) c.setViewOffset(1000 * aspect, 1000, 0, 1000 * yoff, 1000 * aspect, 1000);
  else c.clearViewOffset();
};
const Rig: React.FC<{cam: Cam; aspect: number; yoff: number}> = ({cam, aspect, yoff}) => {
  const {camera} = useThree();
  const c = camera as THREE.PerspectiveCamera;
  c.fov = Math.min(100, vfovOf(cam.hfov, aspect));
  c.aspect = aspect; c.near = 0.1; c.far = 400;
  c.position.copy(cam.pos); c.lookAt(cam.look); shiftView(c, aspect, yoff); c.updateProjectionMatrix();
  return null;
};

// ---------------------------------------------------------------- 3D pieces
const cloudVS = /* glsl */ `
attribute vec2 px; attribute float d;
uniform float uFx, uFy, uCx, uCy, uW, uH, uFar, uStep, uProj, uTop;
varying vec2 vUv; varying float vKeep;
void main() {
  vUv = vec2(px.x / uW, 1.0 - px.y / uH);
  vec3 p = vec3((px.x - uCx) / uFx * d, -(px.y - uCy) / uFy * d, -d);
  vKeep = (d > 0.8 && d < uFar && p.y < uTop) ? 1.0 : 0.0;
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = clamp(1.6 * d * uStep / uFx * uProj / -mv.z, 1.0, 9.0);
}`;
const cloudFS = /* glsl */ `
uniform sampler2D uTex; uniform float uOpacity;
varying vec2 vUv; varying float vKeep;
void main() {
  if (vKeep < 0.5) discard;
  vec2 c = gl_PointCoord - 0.5; if (dot(c, c) > 0.25) discard;
  gl_FragColor = vec4(texture2D(uTex, vUv).rgb * 1.06, uOpacity);
}`;
const Cloud: React.FC<{a: Assets; k: number; opacity: number; proj: number; far: number; top: number}> = ({a, k, opacity, proj, far, top}) => {
  const [gw, gh] = D.grid;
  const tex = a.photos[k];
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const px = new Float32Array(gw * gh * 2);
    for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
      px[2 * (j * gw + i)] = D.grid_step / 2 + i * D.grid_step;
      px[2 * (j * gw + i) + 1] = D.grid_step / 2 + j * D.grid_step;
    }
    g.setAttribute('px', new THREE.BufferAttribute(px, 2));
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(gw * gh * 3), 3));
    g.setAttribute('d', new THREE.BufferAttribute(new Float32Array(gw * gh), 1));
    return g;
  }, [gw, gh]);
  useMemo(() => {
    const at = geom.getAttribute('d') as THREE.BufferAttribute;
    const arr = at.array as Float32Array;
    const off = k * gw * gh;
    for (let i = 0; i < gw * gh; i++) arr[i] = a.depth[off + i] / 100;
    at.needsUpdate = true;
  }, [geom, a, k, gw, gh]);
  const mat = useMemo(() => new THREE.ShaderMaterial({
    vertexShader: cloudVS, fragmentShader: cloudFS, transparent: true, depthWrite: true,
    uniforms: {uTex: {value: null}, uOpacity: {value: 1}, uFx: {value: FX}, uFy: {value: FY}, uCx: {value: CX}, uCy: {value: CY},
      uW: {value: IW}, uH: {value: IH}, uFar: {value: 70}, uStep: {value: D.grid_step}, uProj: {value: 1}, uTop: {value: 4.5}},
  }), []);
  mat.uniforms.uTex.value = tex;
  mat.uniforms.uOpacity.value = opacity;
  mat.uniforms.uProj.value = proj;
  mat.uniforms.uFar.value = far;
  mat.uniforms.uTop.value = top;
  return <points geometry={geom} material={mat} frustumCulled={false} />;
};

const lidarVS = /* glsl */ `
uniform float uReveal, uProj;
varying float vA;
void main() {
  float r = length(position.xz);
  vA = 1.0 - smoothstep(uReveal - 4.0, uReveal, r);
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = clamp(0.09 * uProj / -mv.z, 1.4, 4.5);
}`;
const lidarFS = /* glsl */ `
uniform float uOpacity;
varying float vA;
void main() {
  if (vA < 0.01) discard;
  vec2 c = gl_PointCoord - 0.5; if (dot(c, c) > 0.25) discard;
  gl_FragColor = vec4(0.93, 0.98, 1.0, uOpacity * vA);
}`;
const Lidar: React.FC<{a: Assets; k: number; reveal: number; opacity: number; proj: number}> = ({a, k, reveal, opacity, proj}) => {
  const geom = useMemo(() => {
    const [o, n] = D.lidar_idx[k];
    const p = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const q = 3 * (o + i);
      p[3 * i] = a.lidar[q] / 100; p[3 * i + 1] = -a.lidar[q + 1] / 100; p[3 * i + 2] = -a.lidar[q + 2] / 100;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    return g;
  }, [a, k]);
  const mat = useMemo(() => new THREE.ShaderMaterial({vertexShader: lidarVS, fragmentShader: lidarFS, transparent: true, depthWrite: false,
    uniforms: {uReveal: {value: 0}, uOpacity: {value: 1}, uProj: {value: 1}}}), []);
  mat.uniforms.uReveal.value = reveal; mat.uniforms.uOpacity.value = opacity; mat.uniforms.uProj.value = proj;
  return <points geometry={geom} material={mat} frustumCulled={false} />;
};

// the road the depth model found: a grid at the estimated ground height that stays put in the world as we drive
// (snapped to its 2 m cells), range rings that ride with our car, and the camera-height line
const Road: React.FC<{ego: EgoPose; grid: number; height: number; s: number}> = ({ego, grid, height, s}) => {
  const d0 = ego.h;
  const gy = ego.pos.y - d0;
  const ahead = new THREE.Vector3(-Math.sin(ego.yaw), 0, -Math.cos(ego.yaw)).multiplyScalar(30).add(ego.pos);
  const snap = (v: number) => Math.round(v / 2) * 2;
  const gridObj = useMemo(() => new THREE.GridHelper(80, 40, ROAD, ROAD), []);
  (gridObj.material as THREE.Material).transparent = true;
  (gridObj.material as THREE.Material).opacity = 0.32 * grid;
  const grow = ramp(s, TL.road + 0.3, TL.road + 1.3);
  return (
    <>
      <primitive object={gridObj} position={[snap(ahead.x), gy + 0.01, snap(ahead.z)]} />
      <group position={[ego.pos.x, gy, ego.pos.z]} rotation={[0, ego.yaw, 0]}>
        {[10, 20, 40].map((r) => (
          <mesh key={r} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.03, 0]}>
            <ringGeometry args={[r - 0.1, r + 0.1, 128, 1, Math.PI * 0.25, Math.PI * 0.5]} />
            <meshBasicMaterial color={ROAD} transparent opacity={0.6 * grid} side={THREE.DoubleSide} />
          </mesh>
        ))}
        {[10, 20, 40].map((r) => (
          <sprite key={`l${r}`} position={[-r * 0.72 - 1.2, 0.5, -r * 0.69]} scale={[2.6, 0.52, 1]}>
            <spriteMaterial map={label(`${r} m`, ROAD)} transparent opacity={0.9 * grid} depthTest={false} />
          </sprite>
        ))}
        {height > 0 && (
          <>
            <mesh position={[0, d0 * (1 - 0.5 * grow), 0]} scale={[1, grow, 1]}>
              <cylinderGeometry args={[0.05, 0.05, d0, 12]} />
              <meshBasicMaterial color="#FFFFFF" transparent opacity={height} />
            </mesh>
            <mesh position={[0, d0, 0]}><sphereGeometry args={[0.16, 20, 14]} /><meshBasicMaterial color="#FFFFFF" transparent opacity={height} /></mesh>
            <mesh position={[0, 0.03, 0]} rotation={[-Math.PI / 2, 0, 0]}><ringGeometry args={[0.3, 0.42, 40]} /><meshBasicMaterial color="#FFFFFF" transparent opacity={height} side={THREE.DoubleSide} /></mesh>
          </>
        )}
      </group>
    </>
  );
};

const EgoCar: React.FC<{ego: EgoPose; opacity: number}> = ({ego, opacity}) => (
  <group position={[ego.pos.x, ego.pos.y - ego.h, ego.pos.z]} rotation={[0, ego.yaw, 0]}>
    <group position={[0, 0.78, 1.7]}>
      <mesh><boxGeometry args={[1.8, 1.25, 4.4]} /><meshBasicMaterial color="#0E1A24" transparent opacity={0.6 * opacity} depthWrite={false} /></mesh>
      <lineSegments><edgesGeometry args={[new THREE.BoxGeometry(1.8, 1.25, 4.4)]} /><lineBasicMaterial color={ROAD} transparent opacity={0.9 * opacity} /></lineSegments>
    </group>
  </group>
);

const EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
// a human 3D label box, corners already in the world
const GTBox: React.FC<{c: THREE.Vector3[]; opacity: number}> = ({c, opacity}) => (
  <group>
    {EDGES.map(([i, j], e) => {
      const a = c[i], b = c[j];
      const len = a.distanceTo(b);
      const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
      return (
        <mesh key={e} position={a.clone().add(b).multiplyScalar(0.5)} quaternion={q}>
          <cylinderGeometry args={[0.045, 0.045, len, 6]} />
          <meshBasicMaterial color="#FFFFFF" transparent opacity={opacity} />
        </mesh>
      );
    })}
  </group>
);

// our answer for one car: a glowing pin standing on the road plane at the estimated centre (p in the world)
type PinData = {track: number; p: THREE.Vector3; dist: number; color: string; drop: number; text: string | null; big: boolean;
  ls: number; cx: number; trail: THREE.Vector3[] | null};
const Pin: React.FC<{d: PinData}> = ({d: {p, color, drop, text, big, ls, cx}}) => {
  const lift = (1 - drop) * 7;
  return (
    <group position={[p.x, p.y + lift, p.z]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.04, 0]}>
        <ringGeometry args={[big ? 0.55 : 0.28, big ? 0.85 : 0.42, 40]} />
        <meshBasicMaterial color={color} transparent opacity={drop} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, 1.0, 0]}><cylinderGeometry args={[0.07, 0.07, 2.0, 10]} /><meshBasicMaterial color={color} transparent opacity={drop} /></mesh>
      <mesh position={[0, 2.05, 0]}><sphereGeometry args={[0.2, 16, 12]} /><meshBasicMaterial color={color} transparent opacity={drop} /></mesh>
      {text && (
        <sprite position={[0, 2.6 + ls * 0.3, 0]} scale={[ls * 5, ls, 1]} center={new THREE.Vector2(cx, 0.5)}>
          <spriteMaterial map={label(text, color)} transparent opacity={drop} depthTest={false} />
        </sprite>
      )}
    </group>
  );
};

const colorOf = (cls: string) => (cls === 'person' ? PERSON : CAR);
const isCar = (cls: string) => cls === 'car' || cls === 'truck' || cls === 'bus';

// labels, nearest car first: a label that would cover a nearer one is hidden, one at the frame edge slides inside
function placeLabels(pins: PinData[], cam: Cam, aspect: number, yoff: number, vfov: number, proj: number, vw: number, vh: number) {
  const pc = new THREE.PerspectiveCamera(vfov, aspect, 0.1, 400);
  pc.position.copy(cam.pos); pc.lookAt(cam.look); pc.updateMatrixWorld(); shiftView(pc, aspect, yoff);
  const kept: number[][] = [];
  const M = 14, GAP = 6;
  for (const d of [...pins].sort((x, y) => x.dist - y.dist)) {
    if (!d.text) continue;
    const anchor = d.p.clone().add(new THREE.Vector3(0, 2.6 + d.ls * 0.3, 0));
    const zc = -anchor.clone().applyMatrix4(pc.matrixWorldInverse).z;
    if (zc < 0.5) { d.text = null; continue; }
    const n = anchor.clone().project(pc);
    const px = ((n.x + 1) / 2) * vw, py = ((1 - n.y) / 2) * vh;
    const sw = (d.ls * 5 * proj) / zc, sh = (d.ls * proj) / zc;
    label(d.text, d.color);
    const bw = sw * (labelFrac.get(d.text + d.color) ?? 1), bh = sh * 0.75;
    let x0 = px - bw / 2, x1 = px + bw / 2;
    const y0 = py - bh / 2, y1 = py + bh / 2;
    const dx = x1 > vw - M ? vw - M - x1 : x0 < M ? M - x0 : 0;
    x0 += dx; x1 += dx;
    if (y0 < 0 || y1 > vh || bw > vw - 2 * M || kept.some(([a0, b0, a1, b1]) => x0 < a1 + GAP && x1 > a0 - GAP && y0 < b1 + GAP && y1 > b0 - GAP)) {
      d.text = null;
      continue;
    }
    kept.push([x0, y0, x1, y1]);
    d.cx = 0.5 - dx / sw;
  }
}

const World: React.FC<{a: Assets; s: number; cam: Cam; ego: EgoPose; aspect: number; vw: number; vh: number; yshift: number}> = ({a, s, cam, ego, aspect, vw, vh, yshift}) => {
  const kf = kAt(s);
  const k = Math.floor(kf);
  const k1 = Math.min(NF - 1, k + 1);
  const fr = D.frames[k];
  const nx = D.frames[k1];
  const al = kf - k;
  const vfov = Math.min(100, vfovOf(cam.hfov, aspect));
  const proj = vh / 2 / Math.tan((vfov * Math.PI) / 360);
  const yoff = yshift * ramp(s, TL.lift, TL.road, INOUT);
  const dim = 1 - 0.62 * win(s, TL.res, TL.end + 1, 0.6);
  const cloudOp = ramp(s, TL.lift, TL.lift + 0.15) * dim * (1 - 0.62 * win(s, TL.proof + 0.2, TL.run + 0.8, 0.8));
  const grid = (ramp(s, TL.road, TL.road + 0.8) * (1 - 0.55 * ramp(s, TL.pins, TL.pins + 1))) * dim;
  const height = win(s, TL.road, TL.pins + 0.4);
  const pinsOn = s >= TL.pins;
  const lidarOp = win(s, TL.proof, TL.run + 1.2, 0.3) * 0.95;
  const reveal = 80 * ramp(s, TL.proof, TL.proof + 2.2, INOUT);
  const gtOp = win(s, TL.proof + 1.6, TL.run + 1.2) * 0.95;
  const running = s >= TL.run;
  // pins: our ground position of every car, moved smoothly in the world between two drive frames
  const pins: PinData[] = [];
  if (pinsOn) fr.objects.filter((o) => isCar(o.cls) || o.cls === 'person').forEach((o, i) => {
    const [x, z] = o.c.hybrid;
    if (z > 75) return;
    const p = onGround(k, x, z);
    let dist = Math.hypot(x, z);
    const o2 = k1 !== k ? nx.objects.find((q) => q.track === o.track) : undefined;
    if (o2) {
      const [x2, z2] = o2.c.hybrid;
      p.lerp(onGround(k1, x2, z2), al);
      dist += (Math.hypot(x2, z2) - dist) * al;
    }
    const [first, last] = SPAN.get(o.track)!;
    const born = first > 0 ? ramp(kf, first, first + 3) : 1;
    const gone = last < NF - 1 && !o2 ? 1 - ramp(kf, last, last + 1, Easing.linear) : 1;
    const drop = (running ? 1 : ramp(s, TL.pins + 0.12 * i, TL.pins + 0.7 + 0.12 * i)) * born * gone;
    const sp = o.kmh;
    const moving = running && sp && sp.hybrid !== undefined && sp.gt !== undefined && (sp.gt > 5 || sp.hybrid > 5);
    const text = s >= TL.res ? null : moving ? `${Math.round(sp!.hybrid)} km/h · true ${Math.round(sp!.gt)}` : (z < 45 ? `${Math.round(dist)} m` : null);
    const trail = moving && o.trail.length > 2 ? o.trail.map(([tx, tz]) => onGround(k, tx, tz).add(new THREE.Vector3(0, 0.06, 0))) : null;
    pins.push({track: o.track, p, dist, color: colorOf(o.cls), drop: drop * dim, text, big: isCar(o.cls), ls: 1.0 + dist * 0.06, cx: 0.5, trail});
  });
  placeLabels(pins, cam, aspect, yoff, vfov, proj, vw, vh);
  // human 3D label boxes, moved smoothly in the world like the pins
  const gts = gtOp > 0 ? fr.gt.map((g) => {
    const g2 = k1 !== k ? nx.gt.find((q) => q.tid === g.tid) : undefined;
    const c = g.corners.map((q, i) => {
      const v = v3(q[0], q[1], q[2]).applyMatrix4(T[k]);
      return g2 ? v.lerp(v3(g2.corners[i][0], g2.corners[i][1], g2.corners[i][2]).applyMatrix4(T[k1]), al) : v;
    });
    return {tid: g.tid, z: g.centre[2], c};
  }) : [];
  return (
    <>
      <color attach="background" args={[BG]} />
      <fog attach="fog" args={[BG, 60, 130]} />
      <ambientLight intensity={0.7} />
      <directionalLight position={[10, 25, 15]} intensity={1.2} />
      <Rig cam={cam} aspect={aspect} yoff={yoff} />
      {/* this frame's depth cloud and laser scan, placed in the world by the car's pose */}
      <group position={TP[k].p} quaternion={TP[k].q}>
        {s >= TL.lift && <Cloud a={a} k={k} opacity={cloudOp * (1 - 0.25 * ramp(s, TL.run, TL.run + 1.5))} proj={proj}
          far={70 - 20 * ramp(s, TL.run, TL.run + 1.5)} top={4.5 - 1.5 * ramp(s, TL.run, TL.run + 1.5)} />}
        {lidarOp > 0 && <Lidar a={a} k={k} reveal={reveal} opacity={lidarOp} proj={proj} />}
      </group>
      {s >= TL.road && <Road ego={ego} grid={grid} height={height} s={s} />}
      {s >= TL.lift + 1.2 && s < TL.pins + 1.5 && <EgoCar ego={ego} opacity={win(s, TL.lift + 1.2, TL.pins + 1.5, 0.8)} />}
      {gts.map((g) => <GTBox key={g.tid} c={g.c} opacity={gtOp * ramp(s, TL.proof + 1.6 + g.z / 40, TL.proof + 2.2 + g.z / 40)} />)}
      {pins.map((d) => (
        <group key={d.track}>
          <Pin d={d} />
          {d.trail && <primitive object={new THREE.Line(new THREE.BufferGeometry().setFromPoints(d.trail),
            new THREE.LineBasicMaterial({color: d.color, transparent: true, opacity: 0.85 * dim}))} />}
        </group>
      ))}
    </>
  );
};

// ---------------------------------------------------------------- 2D layer
const Txt: React.FC<{s: number; at: number; until: number; size: number; color?: string; font?: string; weight?: number;
  style?: CSSProperties; children: React.ReactNode}> = ({s, at, until, size, color = TEXT, font = SANS, weight = 800, style, children}) => {
  const o = win(s, at, until);
  if (o <= 0) return null;
  return (
    <div style={{position: 'absolute', fontFamily: font, fontWeight: weight, fontSize: size, lineHeight: 1.04, color,
      letterSpacing: font === SANS ? -size * 0.025 : 1, opacity: o, transform: `translateY(${(1 - ramp(s, at, at + 0.6)) * 26}px)`, ...style}}>
      {children}
    </div>
  );
};

export const M3D: React.FC = () => {
  const f = useCurrentFrame();
  const {width: W, height: H} = useVideoConfig();
  const t = f / FPS;
  const s = t < COVER ? COVER_AT : t - COVER;
  const a = useAssets();
  const tall = H / W > 1.5;
  const stripH = Math.round((W * IH) / IW);
  const textZone = tall ? 470 : 0;
  const vh = H - stripH - textZone;
  const aspect = W / vh;
  const kf = kAt(s);
  const k = Math.floor(kf);
  const ego = egoAt(kf);
  const cam = toWorld(camAt(s), ego);
  const M = 60;  // side margin
  const big = tall ? 112 : 96;
  const bottom = tall ? 120 : 70;
  const stripDim = 1 - 0.55 * win(s, TL.res, TL.end + 1, 0.6);
  return (
    <AbsoluteFill style={{background: BG}}>
      {/* the camera: YOLO26-seg masks on the real frame */}
      <div style={{position: 'absolute', left: 0, top: 0, width: W, height: stripH, opacity: stripDim}}>
        <Img src={staticFile(`seg/${pad(k)}.jpg`)} style={{width: W, height: stripH, display: 'block'}} />
        <div style={{position: 'absolute', left: 16, bottom: 12, fontFamily: MONO, fontSize: 19, color: '#fff', background: 'rgba(5,7,12,0.7)', padding: '4px 10px', borderRadius: 6}}>
          {s < TL.lift ? 'CAMERA · YOLO26x-seg · TensorRT FP16' : 'CAMERA · KITTI drive 0009 (test)'}
        </div>
      </div>
      {/* the 3D world */}
      <div style={{position: 'absolute', left: 0, top: stripH, width: W, height: vh}}>
        {a && (
          <ThreeCanvas width={W} height={vh} camera={{fov: 60, near: 0.1, far: 400, position: [0, 0, 0]}} gl={{antialias: true}}>
            <World a={a} s={s} cam={cam} ego={ego} aspect={aspect} vw={W} vh={vh} yshift={tall ? 0 : 0.14} />
          </ThreeCanvas>
        )}
        <div style={{position: 'absolute', left: 0, right: 0, bottom: 0, height: tall ? 160 : 420, background: `linear-gradient(transparent, ${BG}${tall ? 'FF' : 'E6'} ${tall ? 100 : 70}%)`,
          opacity: s < TL.res ? 1 : 0.6}} />
        <div style={{position: 'absolute', left: 0, right: 0, top: 0, height: 4, background: `linear-gradient(90deg, ${CAR}, ${ROAD})`, opacity: ramp(s, TL.lift, TL.lift + 0.5)}} />
      </div>

      {/* statements: one short line per beat */}
      <div style={{position: 'absolute', left: M, right: M, bottom}}>
        <Txt s={s} at={0.15} until={TL.lift - 0.05} size={big * 1.25} style={{position: 'relative'}}>One camera.</Txt>
        <Txt s={s} at={0.9} until={TL.lift - 0.05} size={big * 1.25} color={CAR} style={{position: 'relative'}}>Every car, in metres.</Txt>
      </div>
      <div style={{position: 'absolute', left: M, right: M, bottom}}>
        <Beat s={s} at={TL.lift + 0.6} until={TL.road} tag="YOLO26x-depth · TensorRT FP16" size={big}>The frame,|lifted into 3D.</Beat>
        <Beat s={s} at={TL.road + 0.2} until={TL.pins} tag="road plane from the depth map" size={big} sub={`camera height ${ego.h.toFixed(2)} m · true ${D.cam_h_true.toFixed(2)} m`}>It finds|the road itself.</Beat>
        <Beat s={s} at={TL.pins + 0.2} until={TL.proof} tag="ground position of every car" size={big}>Each car:|a point in metres.</Beat>
        <Beat s={s} at={TL.proof + 0.3} until={TL.run} tag="white: LiDAR laser · boxes: human 3D labels" size={big}>Checked against|a laser scanner.</Beat>
        <Beat s={s} at={TL.run + 1.6} until={TL.res} tag="km/h from 3 s of positions · true = labels" size={big}>Speed from|the same points.</Beat>
      </div>

      <Results s={s} W={W} H={H} tall={tall} />
      <Fail s={s} W={W} H={H} tall={tall} />
      <Outro s={s} W={W} H={H} tall={tall} />
    </AbsoluteFill>
  );
};

const Beat: React.FC<{s: number; at: number; until: number; tag: string; size: number; sub?: string; children: string}> = ({s, at, until, tag, size, sub, children}) => (
  <div style={{position: 'absolute', left: 0, right: 0, bottom: 0}}>
    <Txt s={s} at={at} until={until} size={24} font={MONO} weight={600} color={ROAD} style={{position: 'relative', marginBottom: 14, letterSpacing: 2}}>{tag.toUpperCase()}</Txt>
    {children.split('|').map((l, i) => (
      <Txt key={i} s={s} at={at + 0.12 * (i + 1)} until={until} size={size} style={{position: 'relative'}}>{l}</Txt>
    ))}
    {sub && <Txt s={s} at={at + 0.9} until={until} size={34} font={MONO} weight={600} style={{position: 'relative', marginTop: 18}}>{sub}</Txt>}
  </div>
);

const Results: React.FC<{s: number; W: number; H: number; tall: boolean}> = ({s, W, H, tall}) => {
  if (s < TL.res - 0.1 || s > TL.fail + 0.1) return null;
  const top = tall ? 560 : 400;
  const b1 = win(s, TL.res, TL.res2), b2 = win(s, TL.res2, TL.fail);
  const huge = tall ? 300 : 260;
  return (
    <AbsoluteFill>
      <div style={{position: 'absolute', left: 60, right: 60, top}}>
        <Txt s={s} at={TL.res + 0.1} until={TL.fail} size={24} font={MONO} weight={600} color={ROAD} style={{position: 'relative', letterSpacing: 2}}>
          {`TEST · ${RES.drives} DRIVES THE MODEL NEVER SAW · ${RES.cars} CARS`}
        </Txt>
        <div style={{position: 'relative', height: huge * 1.9, marginTop: 20}}>
          <div style={{position: 'absolute', inset: 0, opacity: b1}}>
            <div style={{fontFamily: SANS, fontWeight: 800, fontSize: huge, color: TEXT, letterSpacing: -huge * 0.04, lineHeight: 1}}>
              {RES.hy}<span style={{fontSize: huge * 0.42, color: CAR}}> m</span>
            </div>
            <div style={{fontFamily: SANS, fontWeight: 600, fontSize: tall ? 52 : 46, color: TEXT, marginTop: 10}}>median error of a car’s position</div>
            <div style={{fontFamily: MONO, fontSize: 30, color: '#B8C0C8', marginTop: 18}}>{`classic camera geometry: ${RES.b0} m`}</div>
          </div>
          <div style={{position: 'absolute', inset: 0, opacity: b2}}>
            <div style={{fontFamily: SANS, fontWeight: 800, fontSize: huge, color: TEXT, letterSpacing: -huge * 0.04, lineHeight: 1}}>
              {RES.kmh}<span style={{fontSize: huge * 0.42, color: CAR}}> km/h</span>
            </div>
            <div style={{fontFamily: SANS, fontWeight: 600, fontSize: tall ? 52 : 46, color: TEXT, marginTop: 10}}>median speed error</div>
            <div style={{fontFamily: MONO, fontSize: 30, color: '#B8C0C8', marginTop: 18}}>{`classic: ${RES.kmhB0} km/h  ·  ${RES.ms} ms per frame`}</div>
            <div style={{fontFamily: MONO, fontSize: 30, color: NV, marginTop: 8}}>TensorRT FP16 · one RTX 5060 Ti</div>
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Fail: React.FC<{s: number; W: number; H: number; tall: boolean}> = ({s, W, H, tall}) => {
  const o = win(s, TL.fail, TL.outro);
  if (o <= 0) return null;
  const iw = W - 120, ih = Math.round((iw * 2) / 3);
  const top = tall ? 360 : 120;
  const inset = Math.round(iw * 0.38);
  return (
    <AbsoluteFill style={{background: BG, opacity: o}}>
      <div style={{position: 'absolute', left: 60, top}}>
        <div style={{fontFamily: MONO, fontWeight: 600, fontSize: 24, color: CAR, letterSpacing: 2}}>WHERE IT BREAKS</div>
        <div style={{fontFamily: SANS, fontWeight: 800, fontSize: tall ? 84 : 76, color: TEXT, lineHeight: 1.04, letterSpacing: -2, marginTop: 12, width: iw}}>
          A car hidden behind another, seen through glass.
        </div>
      </div>
      <div style={{position: 'absolute', left: 60, top: top + (tall ? 330 : 300), width: iw, height: ih, borderRadius: 16, overflow: 'hidden'}}>
        <Img src={staticFile('fail_photo.jpg')} style={{width: iw, height: ih, display: 'block'}} />
        <div style={{position: 'absolute', right: 16, bottom: 16, width: inset, height: Math.round(inset * 2 / 3), borderRadius: 10, overflow: 'hidden', border: '3px solid #fff',
          opacity: ramp(s, TL.fail + 0.9, TL.fail + 1.4)}}>
          <Img src={staticFile('fail_depth.jpg')} style={{width: '100%', height: '100%', display: 'block'}} />
        </div>
      </div>
      <div style={{position: 'absolute', left: 60, top: top + (tall ? 330 : 300) + ih + 28, fontFamily: MONO, fontSize: 32, color: TEXT, opacity: ramp(s, TL.fail + 1.2, TL.fail + 1.8)}}>
        <span style={{color: DIM}}>laser</span> 19 m   <span style={{color: DIM}}>depth</span> <span style={{color: CAR}}>37 m</span>
      </div>
    </AbsoluteFill>
  );
};

const Outro: React.FC<{s: number; W: number; H: number; tall: boolean}> = ({s, W, H, tall}) => {
  const o = ramp(s, TL.outro, TL.outro + 0.5);
  if (o <= 0) return null;
  const size = tall ? 92 : 84;
  return (
    <AbsoluteFill style={{background: BG, opacity: o}}>
      <div style={{position: 'absolute', left: 60, right: 60, top: tall ? 620 : 380}}>
        <div style={{fontFamily: SANS, fontWeight: 800, fontSize: size, color: TEXT, lineHeight: 1.04, letterSpacing: -2}}>
          One camera.<br /><span style={{color: CAR}}>Metres you can check.</span>
        </div>
        <div style={{fontFamily: MONO, fontSize: 30, color: TEXT, marginTop: 44}}>Ultralytics YOLO26 seg + depth</div>
        <div style={{fontFamily: MONO, fontSize: 30, color: NV, marginTop: 8}}>NVIDIA TensorRT FP16 · RTX 5060 Ti</div>
        <div style={{fontFamily: MONO, fontSize: 30, color: ROAD, marginTop: 44}}>{REPO}</div>
        <div style={{fontFamily: MONO, fontSize: 22, color: DIM, marginTop: 40, lineHeight: 1.5}}>
          pre-registered test · run once · all numbers published<br />data: KITTI raw (Geiger et al.) · CC BY-NC-SA 3.0
        </div>
      </div>
    </AbsoluteFill>
  );
};
