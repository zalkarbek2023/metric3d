import React from 'react';
import {Composition} from 'remotion';
import {M3D, TOTAL, FPS} from './M3D';

// one scene, two layouts: 4:5 for X/LinkedIn, 9:16 for YouTube Shorts
export const Root: React.FC = () => (
  <>
    <Composition id="M45" component={M3D} durationInFrames={TOTAL} fps={FPS} width={1080} height={1350} />
    <Composition id="M916" component={M3D} durationInFrames={TOTAL} fps={FPS} width={1080} height={1920} />
  </>
);
