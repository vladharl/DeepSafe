// frontend/src/components/VideoTimeline.js
import React, { useState, useMemo } from 'react';
import { Clock, AlertTriangle, CheckCircle, ChevronLeft, ChevronRight } from 'lucide-react';
import { formatProbability } from '../utils/formatters';

/**
 * VideoTimeline - Displays a timeline visualization of frame-by-frame analysis
 * Shows color-coded markers for each analyzed frame with their scores
 */
const VideoTimeline = ({
  frameAnalysis,
  temporalAnalysis,
  onFrameSelect,
  videoDuration
}) => {
  const [selectedFrameIndex, setSelectedFrameIndex] = useState(null);
  const [hoveredFrame, setHoveredFrame] = useState(null);

  // Extract frames from frameAnalysis
  const frames = frameAnalysis?.frames || [];
  const suspiciousFrames = frameAnalysis?.suspicious_frames || [];
  const totalDuration = videoDuration || frameAnalysis?.video_duration_seconds || 0;

  // Calculate frame positions and colors
  const frameMarkers = useMemo(() => {
    if (!frames.length) return [];

    return frames.map((frame, index) => {
      const position = totalDuration > 0
        ? (frame.timestamp_seconds / totalDuration) * 100
        : (index / frames.length) * 100;

      const score = frame.aggregate_frame_score;
      const isSuspicious = suspiciousFrames.some(sf => sf.frame_index === frame.frame_index);

      // Determine color based on score
      let colorClass = 'bg-neutral-400'; // Unknown
      if (score !== null && score !== undefined) {
        if (score >= 0.7) {
          colorClass = 'bg-danger-500 hover:bg-danger-600'; // Fake
        } else if (score >= 0.4) {
          colorClass = 'bg-warning-500 hover:bg-warning-600'; // Uncertain
        } else {
          colorClass = 'bg-success-500 hover:bg-success-600'; // Real
        }
      }

      return {
        ...frame,
        index,
        position,
        colorClass,
        isSuspicious,
        score
      };
    });
  }, [frames, suspiciousFrames, totalDuration]);

  const handleFrameClick = (frame) => {
    setSelectedFrameIndex(frame.index);
    if (onFrameSelect) {
      onFrameSelect(frame);
    }
  };

  const formatTimestamp = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (!frames.length) {
    return (
      <div className="bg-neutral-50 dark:bg-neutral-700/50 rounded-lg p-6 text-center">
        <p className="text-neutral-500 dark:text-neutral-400">
          No frame analysis data available
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Stats summary */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-neutral-50 dark:bg-neutral-700/50 rounded-lg p-4 text-center">
          <p className="text-2xl font-bold text-neutral-800 dark:text-neutral-100">
            {frameAnalysis?.total_frames_analyzed || frames.length}
          </p>
          <p className="text-xs text-neutral-500 dark:text-neutral-400">Frames Analyzed</p>
        </div>
        <div className="bg-neutral-50 dark:bg-neutral-700/50 rounded-lg p-4 text-center">
          <p className="text-2xl font-bold text-neutral-800 dark:text-neutral-100">
            {frameAnalysis?.frame_interval_seconds || 1.0}s
          </p>
          <p className="text-xs text-neutral-500 dark:text-neutral-400">Interval</p>
        </div>
        <div className="bg-neutral-50 dark:bg-neutral-700/50 rounded-lg p-4 text-center">
          <p className="text-2xl font-bold text-danger-600 dark:text-danger-400">
            {suspiciousFrames.length}
          </p>
          <p className="text-xs text-neutral-500 dark:text-neutral-400">Suspicious Frames</p>
        </div>
        <div className="bg-neutral-50 dark:bg-neutral-700/50 rounded-lg p-4 text-center">
          <p className="text-2xl font-bold text-neutral-800 dark:text-neutral-100">
            {formatTimestamp(totalDuration)}
          </p>
          <p className="text-xs text-neutral-500 dark:text-neutral-400">Duration</p>
        </div>
      </div>

      {/* Timeline */}
      <div className="bg-neutral-50 dark:bg-neutral-700/50 rounded-lg p-6">
        <h3 className="text-sm font-semibold mb-4 text-neutral-800 dark:text-neutral-100 flex items-center">
          <Clock className="h-5 w-5 mr-2 text-neutral-500 dark:text-neutral-400" />
          Frame Timeline
        </h3>

        {/* Timeline bar */}
        <div className="relative">
          {/* Background track */}
          <div className="h-8 bg-gradient-to-r from-success-100 via-warning-100 to-danger-100 dark:from-success-900/30 dark:via-warning-900/30 dark:to-danger-900/30 rounded-full relative overflow-visible">
            {/* Frame markers */}
            {frameMarkers.map((frame) => (
              <div
                key={frame.frame_index}
                className={`absolute top-0 w-3 h-8 rounded-sm cursor-pointer transition-all duration-200
                  ${frame.colorClass}
                  ${selectedFrameIndex === frame.index ? 'ring-2 ring-primary-500 ring-offset-2 scale-110 z-10' : ''}
                  ${frame.isSuspicious ? 'animate-pulse' : ''}`}
                style={{ left: `calc(${frame.position}% - 6px)` }}
                onClick={() => handleFrameClick(frame)}
                onMouseEnter={() => setHoveredFrame(frame)}
                onMouseLeave={() => setHoveredFrame(null)}
              >
                {/* Tooltip */}
                {hoveredFrame?.frame_index === frame.frame_index && (
                  <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 z-20">
                    <div className="bg-neutral-800 dark:bg-neutral-900 text-white px-3 py-2 rounded-lg text-xs whitespace-nowrap shadow-lg">
                      <p className="font-medium">{formatTimestamp(frame.timestamp_seconds)}</p>
                      <p className="text-neutral-300">
                        Score: {frame.score !== null ? formatProbability(frame.score) : 'N/A'}
                      </p>
                      {frame.isSuspicious && (
                        <p className="text-danger-400 flex items-center mt-1">
                          <AlertTriangle size={12} className="mr-1" /> Suspicious
                        </p>
                      )}
                    </div>
                    <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-neutral-800 dark:border-t-neutral-900" />
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Time labels */}
          <div className="flex justify-between mt-2 text-xs text-neutral-500 dark:text-neutral-400">
            <span>0:00</span>
            <span>{formatTimestamp(totalDuration)}</span>
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center justify-center gap-6 mt-4 text-xs">
          <div className="flex items-center">
            <div className="w-3 h-3 rounded bg-success-500 mr-2" />
            <span className="text-neutral-600 dark:text-neutral-400">Authentic (&lt;40%)</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded bg-warning-500 mr-2" />
            <span className="text-neutral-600 dark:text-neutral-400">Uncertain (40-70%)</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded bg-danger-500 mr-2" />
            <span className="text-neutral-600 dark:text-neutral-400">Fake (&gt;70%)</span>
          </div>
        </div>
      </div>

      {/* Selected frame details */}
      {selectedFrameIndex !== null && frameMarkers[selectedFrameIndex] && (
        <div className="bg-white dark:bg-neutral-700/50 rounded-lg border border-neutral-200 dark:border-neutral-600 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-neutral-800 dark:text-neutral-100">
              Frame at {formatTimestamp(frameMarkers[selectedFrameIndex].timestamp_seconds)}
            </h3>
            <div className="flex gap-2">
              <button
                onClick={() => setSelectedFrameIndex(Math.max(0, selectedFrameIndex - 1))}
                disabled={selectedFrameIndex === 0}
                className="p-1 rounded hover:bg-neutral-100 dark:hover:bg-neutral-600 disabled:opacity-50"
              >
                <ChevronLeft size={20} />
              </button>
              <button
                onClick={() => setSelectedFrameIndex(Math.min(frameMarkers.length - 1, selectedFrameIndex + 1))}
                disabled={selectedFrameIndex === frameMarkers.length - 1}
                className="p-1 rounded hover:bg-neutral-100 dark:hover:bg-neutral-600 disabled:opacity-50"
              >
                <ChevronRight size={20} />
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Thumbnail */}
            {frameMarkers[selectedFrameIndex].thumbnail_base64 && (
              <div className="aspect-video bg-neutral-100 dark:bg-neutral-800 rounded-lg overflow-hidden">
                <img
                  src={`data:image/jpeg;base64,${frameMarkers[selectedFrameIndex].thumbnail_base64}`}
                  alt={`Frame at ${formatTimestamp(frameMarkers[selectedFrameIndex].timestamp_seconds)}`}
                  className="w-full h-full object-contain"
                />
              </div>
            )}

            {/* Model results for this frame */}
            <div className="space-y-3">
              <h4 className="text-xs font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
                Image Model Results
              </h4>
              {frameMarkers[selectedFrameIndex].image_model_results &&
                Object.entries(frameMarkers[selectedFrameIndex].image_model_results).map(([modelId, result]) => (
                  <div key={modelId} className="flex items-center justify-between p-3 bg-neutral-50 dark:bg-neutral-800 rounded-lg">
                    <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
                      {modelId.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                    </span>
                    {result.error ? (
                      <span className="text-xs text-danger-500">Error</span>
                    ) : (
                      <div className="flex items-center gap-2">
                        <span className={`text-sm font-bold ${
                          result.probability >= 0.5 ? 'text-danger-600' : 'text-success-600'
                        }`}>
                          {formatProbability(result.probability)}
                        </span>
                        {result.class === 'fake' ? (
                          <AlertTriangle size={16} className="text-danger-500" />
                        ) : (
                          <CheckCircle size={16} className="text-success-500" />
                        )}
                      </div>
                    )}
                  </div>
                ))
              }

              {/* Aggregate score */}
              <div className="pt-3 border-t border-neutral-200 dark:border-neutral-600">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-semibold text-neutral-800 dark:text-neutral-100">
                    Aggregate Frame Score
                  </span>
                  <span className={`text-lg font-bold ${
                    frameMarkers[selectedFrameIndex].score >= 0.5 ? 'text-danger-600' : 'text-success-600'
                  }`}>
                    {frameMarkers[selectedFrameIndex].score !== null
                      ? formatProbability(frameMarkers[selectedFrameIndex].score)
                      : 'N/A'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Suspicious frames list */}
      {suspiciousFrames.length > 0 && (
        <div className="bg-danger-50 dark:bg-danger-900/20 rounded-lg border border-danger-200 dark:border-danger-800 p-6">
          <h3 className="text-sm font-semibold mb-4 text-danger-800 dark:text-danger-200 flex items-center">
            <AlertTriangle className="h-5 w-5 mr-2" />
            Suspicious Frames Detected
          </h3>
          <div className="space-y-2">
            {suspiciousFrames.slice(0, 5).map((sf, idx) => (
              <button
                key={idx}
                onClick={() => {
                  const frameIdx = frames.findIndex(f => f.frame_index === sf.frame_index);
                  if (frameIdx >= 0) setSelectedFrameIndex(frameIdx);
                }}
                className="w-full flex items-center justify-between p-3 bg-white dark:bg-neutral-800 rounded-lg hover:bg-neutral-50 dark:hover:bg-neutral-700 transition-colors text-left"
              >
                <div>
                  <p className="text-sm font-medium text-neutral-800 dark:text-neutral-200">
                    {formatTimestamp(sf.timestamp_seconds)}
                  </p>
                  <p className="text-xs text-neutral-500 dark:text-neutral-400">
                    {sf.reason}
                  </p>
                </div>
                <span className="text-danger-600 dark:text-danger-400 font-bold">
                  {formatProbability(sf.score)}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoTimeline;
