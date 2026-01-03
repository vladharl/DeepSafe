// frontend/src/components/HeatmapOverlay.js
import React, { useState, useRef, useEffect } from 'react';
import { Eye, EyeOff, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';

/**
 * HeatmapOverlay - Displays localization heatmap overlaid on an image
 * Used to show manipulation probability regions from models like TruFor
 */
const HeatmapOverlay = ({
  originalImage, // Base64 encoded original image
  heatmapImage,  // Base64 encoded heatmap image
  probability,   // Overall fake probability
  modelName,     // Name of the model that generated the heatmap
}) => {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [opacity, setOpacity] = useState(0.5);
  const [zoom, setZoom] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);

  const handleMouseDown = (e) => {
    if (zoom > 1) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
    }
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleZoomIn = () => {
    setZoom((prev) => Math.min(prev + 0.5, 4));
  };

  const handleZoomOut = () => {
    setZoom((prev) => {
      const newZoom = Math.max(prev - 0.5, 1);
      if (newZoom === 1) {
        setPosition({ x: 0, y: 0 });
      }
      return newZoom;
    });
  };

  const handleReset = () => {
    setZoom(1);
    setPosition({ x: 0, y: 0 });
    setOpacity(0.5);
    setShowHeatmap(true);
  };

  useEffect(() => {
    const handleGlobalMouseUp = () => setIsDragging(false);
    window.addEventListener('mouseup', handleGlobalMouseUp);
    return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
  }, []);

  if (!originalImage && !heatmapImage) {
    return (
      <div className="bg-neutral-50 dark:bg-neutral-700/50 rounded-lg p-6 text-center">
        <p className="text-neutral-500 dark:text-neutral-400">
          No heatmap data available
        </p>
        <p className="text-xs text-neutral-400 dark:text-neutral-500 mt-1">
          Heatmap visualization requires TruFor or similar localization-capable model
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-neutral-50 dark:bg-neutral-700/50 rounded-lg p-4">
        <div className="flex items-center gap-4">
          {/* Toggle heatmap */}
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
              showHeatmap
                ? 'bg-primary-100 text-primary-700 dark:bg-primary-900/30 dark:text-primary-300'
                : 'bg-neutral-200 text-neutral-600 dark:bg-neutral-600 dark:text-neutral-300'
            }`}
          >
            {showHeatmap ? <Eye size={16} /> : <EyeOff size={16} />}
            {showHeatmap ? 'Hide Heatmap' : 'Show Heatmap'}
          </button>

          {/* Opacity slider */}
          {showHeatmap && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-neutral-600 dark:text-neutral-400">Opacity:</span>
              <input
                type="range"
                min="0"
                max="100"
                value={opacity * 100}
                onChange={(e) => setOpacity(e.target.value / 100)}
                className="w-24 h-2 bg-neutral-200 dark:bg-neutral-600 rounded-lg appearance-none cursor-pointer"
              />
              <span className="text-sm text-neutral-500 dark:text-neutral-400 w-8">
                {Math.round(opacity * 100)}%
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Zoom controls */}
          <button
            onClick={handleZoomOut}
            disabled={zoom <= 1}
            className="p-2 rounded-lg bg-neutral-200 dark:bg-neutral-600 text-neutral-600 dark:text-neutral-300 hover:bg-neutral-300 dark:hover:bg-neutral-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ZoomOut size={16} />
          </button>
          <span className="text-sm text-neutral-600 dark:text-neutral-400 w-12 text-center">
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={handleZoomIn}
            disabled={zoom >= 4}
            className="p-2 rounded-lg bg-neutral-200 dark:bg-neutral-600 text-neutral-600 dark:text-neutral-300 hover:bg-neutral-300 dark:hover:bg-neutral-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ZoomIn size={16} />
          </button>
          <button
            onClick={handleReset}
            className="p-2 rounded-lg bg-neutral-200 dark:bg-neutral-600 text-neutral-600 dark:text-neutral-300 hover:bg-neutral-300 dark:hover:bg-neutral-500 ml-2"
            title="Reset view"
          >
            <RotateCcw size={16} />
          </button>
        </div>
      </div>

      {/* Image container */}
      <div
        ref={containerRef}
        className="relative overflow-hidden bg-neutral-100 dark:bg-neutral-800 rounded-lg cursor-move"
        style={{ height: '400px' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <div
          className="absolute inset-0 flex items-center justify-center transition-transform duration-150"
          style={{
            transform: `scale(${zoom}) translate(${position.x / zoom}px, ${position.y / zoom}px)`,
          }}
        >
          {/* Original image */}
          {originalImage && (
            <img
              src={originalImage.startsWith('data:') ? originalImage : `data:image/jpeg;base64,${originalImage}`}
              alt="Original"
              className="max-w-full max-h-full object-contain pointer-events-none"
              draggable={false}
            />
          )}

          {/* Heatmap overlay */}
          {showHeatmap && heatmapImage && (
            <img
              src={heatmapImage.startsWith('data:') ? heatmapImage : `data:image/png;base64,${heatmapImage}`}
              alt="Heatmap overlay"
              className="absolute inset-0 max-w-full max-h-full object-contain pointer-events-none"
              style={{ opacity }}
              draggable={false}
            />
          )}
        </div>

        {/* Zoom hint */}
        {zoom === 1 && (
          <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-black/50 text-white text-xs px-3 py-1 rounded-full pointer-events-none">
            Use zoom controls to inspect details
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="bg-neutral-50 dark:bg-neutral-700/50 rounded-lg p-4">
        <h4 className="text-sm font-medium text-neutral-800 dark:text-neutral-100 mb-3">
          Heatmap Legend
        </h4>
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <div className="h-4 rounded-full bg-gradient-to-r from-blue-500 via-green-500 via-yellow-500 via-orange-500 to-red-500" />
            <div className="flex justify-between mt-1 text-xs text-neutral-500 dark:text-neutral-400">
              <span>Authentic</span>
              <span>Low</span>
              <span>Medium</span>
              <span>High</span>
              <span>Manipulated</span>
            </div>
          </div>
        </div>
        {modelName && (
          <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-3">
            Generated by: {modelName}
          </p>
        )}
        {probability !== undefined && probability !== null && (
          <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
            Overall fake probability: {Math.round(probability * 100)}%
          </p>
        )}
      </div>

      {/* Info */}
      <div className="text-xs text-neutral-500 dark:text-neutral-400 bg-neutral-50 dark:bg-neutral-700/50 rounded-lg p-4">
        <p className="font-medium mb-1">About this visualization:</p>
        <p>
          The heatmap shows regions of the image where manipulation is detected.
          Warmer colors (red, orange) indicate higher probability of manipulation,
          while cooler colors (blue, green) suggest authentic regions.
        </p>
      </div>
    </div>
  );
};

export default HeatmapOverlay;
