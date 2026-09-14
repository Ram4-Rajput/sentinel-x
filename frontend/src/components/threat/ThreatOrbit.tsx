import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import * as THREE from "three";
import { useSentinel } from "@/store";
import { riskColor } from "@/lib/format";

// A distinctive 3D threat visualization. Per spec, RISK / UNCERTAINTY /
// NOVELTY are shown as SEPARATE channels — never merged into one gauge:
//
//   RISK        -> the orbiting core body; its radius/speed encode forecast
//                  risk; a marker orbits each forecast horizon (t+1..t+K).
//   UNCERTAINTY -> a translucent halo whose thickness = predictive std.
//   NOVELTY/OOD -> a spiky outer ring that ignites when Mahalanobis novelty
//                  crosses its in-distribution threshold.
//
// Everything is bound to live API values (no decorative animation).

function RiskCore() {
  const forecast = useSentinel((s) => s.forecast);
  const meshRef = useRef<THREE.Mesh>(null);
  const risk = forecast?.risk.risk_now ?? 0;

  useFrame((_, dt) => {
    if (meshRef.current) meshRef.current.rotation.y += dt * (0.2 + risk);
  });

  const color = new THREE.Color(riskColor(risk));
  return (
    <mesh ref={meshRef}>
      <icosahedronGeometry args={[0.55 + risk * 0.25, 1]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={0.4 + risk * 1.6}
        wireframe
      />
    </mesh>
  );
}

// Orbiting markers = forecast risk per horizon t+1..t+K.
function ForecastOrbit() {
  const forecast = useSentinel((s) => s.forecast);
  const groupRef = useRef<THREE.Group>(null);
  const risks = forecast?.risk.forecast_risk ?? [];

  useFrame((_, dt) => {
    if (groupRef.current) groupRef.current.rotation.z += dt * 0.15;
  });

  return (
    <group ref={groupRef}>
      {risks.map((r, i) => {
        const radius = 1.1 + i * 0.28;
        const angle = (i / risks.length) * Math.PI * 2;
        const pos: [number, number, number] = [
          Math.cos(angle) * radius,
          Math.sin(angle) * radius,
          0,
        ];
        const ringPts = Array.from({ length: 65 }, (_, k) => {
          const a = (k / 64) * Math.PI * 2;
          return new THREE.Vector3(Math.cos(a) * radius, Math.sin(a) * radius, 0);
        });
        return (
          <group key={i}>
            <Line points={ringPts} color="#25325a" lineWidth={0.6} />
            <mesh position={pos}>
              <sphereGeometry args={[0.05 + r * 0.12, 16, 16]} />
              <meshStandardMaterial
                color={riskColor(r)}
                emissive={riskColor(r)}
                emissiveIntensity={0.6 + r * 2}
              />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}

// UNCERTAINTY halo — translucent sphere whose opacity/scale = predictive std.
function UncertaintyHalo() {
  const forecast = useSentinel((s) => s.forecast);
  const std = forecast?.uncertainty?.uncertainty_std ?? 0;
  const meshRef = useRef<THREE.Mesh>(null);
  useFrame((state) => {
    if (meshRef.current) {
      const s = 1.6 + std * 3 + Math.sin(state.clock.elapsedTime) * 0.04;
      meshRef.current.scale.setScalar(s);
    }
  });
  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[1, 32, 32]} />
      <meshBasicMaterial
        color="#a970ff"
        transparent
        opacity={0.05 + std * 0.6}
        side={THREE.BackSide}
      />
    </mesh>
  );
}

// NOVELTY/OOD spiky ring — ignites when novelty crosses threshold.
function NoveltyRing() {
  const forecast = useSentinel((s) => s.forecast);
  const nov = forecast?.novelty;
  const groupRef = useRef<THREE.Group>(null);
  const score = nov?.novelty_score ?? 0;
  const threshold = nov?.threshold ?? 3;
  const active = nov?.is_novel ?? false;
  const norm = Math.min(1.5, score / Math.max(0.01, threshold));

  useFrame((_, dt) => {
    if (groupRef.current) groupRef.current.rotation.x += dt * 0.1;
  });

  const spikes = useMemo(() => Array.from({ length: 40 }, (_, i) => i), []);
  const color = active ? "#ffb020" : "#3a4a6a";
  return (
    <group ref={groupRef} rotation={[Math.PI / 2.3, 0, 0]}>
      {spikes.map((i) => {
        const a = (i / spikes.length) * Math.PI * 2;
        const base = 2.1;
        const len = 0.08 + norm * 0.5 * (active ? 1 : 0.3);
        const x = Math.cos(a) * base;
        const y = Math.sin(a) * base;
        return (
          <mesh key={i} position={[x, y, 0]} rotation={[0, 0, a]}>
            <boxGeometry args={[len, 0.02, 0.02]} />
            <meshStandardMaterial
              color={color}
              emissive={color}
              emissiveIntensity={active ? 1.4 : 0.2}
            />
          </mesh>
        );
      })}
    </group>
  );
}

export function ThreatOrbit() {
  return (
    <div className="relative h-full w-full">
      <Canvas camera={{ position: [0, 0, 6], fov: 45 }} dpr={[1, 2]}>
        <ambientLight intensity={0.4} />
        <pointLight position={[5, 5, 5]} intensity={1.2} />
        <RiskCore />
        <ForecastOrbit />
        <UncertaintyHalo />
        <NoveltyRing />
        <EffectComposer>
          <Bloom intensity={0.9} luminanceThreshold={0.2} mipmapBlur />
        </EffectComposer>
        <OrbitControls enablePan={false} enableZoom={false} autoRotate autoRotateSpeed={0.4} />
      </Canvas>
    </div>
  );
}
