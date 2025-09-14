import React, { useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Box } from '@react-three/drei';
import * as THREE from 'three';

function RotatingCube() {
  const meshRef = useRef<THREE.Mesh>(null!);

  useFrame((state, delta) => {
    meshRef.current.rotation.x += delta;
    meshRef.current.rotation.y += delta * 0.5;
  });

  return (
    <Box ref={meshRef} args={[2, 2, 2]}>
      <meshStandardMaterial color="#4f46e5" />
    </Box>
  );
}

function Model3D() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12">
      <div className="container mx-auto px-4">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            3D Модель
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Интерактивная 3D модель для изучения жестового языка
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-xl p-6 max-w-4xl mx-auto">
          <div className="h-96 w-full">
            <Canvas camera={{ position: [0, 0, 5] }}>
              <ambientLight intensity={0.5} />
              <pointLight position={[10, 10, 10]} />
              <RotatingCube />
              <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
            </Canvas>
          </div>
          
          <div className="mt-6 text-center">
            <p className="text-gray-600">
              Используйте мышь для поворота, масштабирования и перемещения 3D модели
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Model3D;