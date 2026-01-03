// frontend/src/components/FaceAnalysisGrid.js
import React, { useState, useMemo } from 'react';
import { User, AlertTriangle, CheckCircle, TrendingUp, TrendingDown } from 'lucide-react';
import { formatProbability } from '../utils/formatters';

/**
 * FaceAnalysisGrid - Displays per-face analysis results from video detection
 * Shows face detection results across all analyzed frames
 */
const FaceAnalysisGrid = ({ temporalAnalysis, frameAnalysis }) => {
  const [selectedFaceId, setSelectedFaceId] = useState(null);
  const [viewMode, setViewMode] = useState('grid'); // 'grid' or 'timeline'

  // Extract face data from temporal analysis (from video model like Cross-Efficient-ViT)
  const faceData = useMemo(() => {
    if (!temporalAnalysis) return { faces: [], frameDetails: [] };

    // Get frame details from the first video model that has them
    let frameDetails = [];
    let perFaceScores = [];

    for (const [modelName, data] of Object.entries(temporalAnalysis)) {
      if (data.frame_details && data.frame_details.length > 0) {
        frameDetails = data.frame_details;
      }
      if (data.per_face_scores && data.per_face_scores.length > 0) {
        perFaceScores = data.per_face_scores;
      }
    }

    // Build face tracking across frames
    const faceMap = new Map();

    frameDetails.forEach((frame) => {
      if (frame.faces && frame.faces.length > 0) {
        frame.faces.forEach((face) => {
          const faceId = face.face_id;
          if (!faceMap.has(faceId)) {
            faceMap.set(faceId, {
              faceId,
              appearances: [],
              scores: [],
            });
          }
          faceMap.get(faceId).appearances.push({
            frameIndex: frame.frame_index,
            timestamp: frame.timestamp_seconds,
            bbox: face.bbox,
            score: face.score,
            confidence: face.confidence,
          });
          faceMap.get(faceId).scores.push(face.score);
        });
      }
    });

    // Calculate aggregate stats for each face
    const faces = Array.from(faceMap.values()).map((face) => ({
      ...face,
      avgScore: face.scores.length > 0
        ? face.scores.reduce((a, b) => a + b, 0) / face.scores.length
        : null,
      maxScore: face.scores.length > 0 ? Math.max(...face.scores) : null,
      minScore: face.scores.length > 0 ? Math.min(...face.scores) : null,
      frameCount: face.appearances.length,
    }));

    return { faces, frameDetails, perFaceScores };
  }, [temporalAnalysis]);

  const { faces, frameDetails } = faceData;

  // Calculate overall face stats
  const faceStats = useMemo(() => {
    if (!faces.length) return null;

    const totalFaces = faces.length;
    const suspiciousFaces = faces.filter((f) => f.avgScore >= 0.5).length;
    const avgScore = faces.reduce((sum, f) => sum + (f.avgScore || 0), 0) / totalFaces;

    return {
      totalFaces,
      suspiciousFaces,
      avgScore,
    };
  }, [faces]);

  const formatTimestamp = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getScoreColor = (score) => {
    if (score === null || score === undefined) return 'text-neutral-500';
    if (score >= 0.7) return 'text-danger-600 dark:text-danger-400';
    if (score >= 0.4) return 'text-warning-600 dark:text-warning-400';
    return 'text-success-600 dark:text-success-400';
  };

  const getScoreBgColor = (score) => {
    if (score === null || score === undefined) return 'bg-neutral-100 dark:bg-neutral-700';
    if (score >= 0.7) return 'bg-danger-50 dark:bg-danger-900/30';
    if (score >= 0.4) return 'bg-warning-50 dark:bg-warning-900/30';
    return 'bg-success-50 dark:bg-success-900/30';
  };

  if (!faces.length) {
    return (
      <div className="bg-neutral-50 dark:bg-neutral-700/50 rounded-lg p-6 text-center">
        <User className="h-12 w-12 mx-auto text-neutral-400 mb-3" />
        <p className="text-neutral-500 dark:text-neutral-400">
          No face detection data available
        </p>
        <p className="text-xs text-neutral-400 dark:text-neutral-500 mt-1">
          Face-level analysis requires video model processing
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Stats summary */}
      {faceStats && (
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-neutral-50 dark:bg-neutral-700/50 rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-neutral-800 dark:text-neutral-100">
              {faceStats.totalFaces}
            </p>
            <p className="text-xs text-neutral-500 dark:text-neutral-400">Faces Detected</p>
          </div>
          <div className="bg-neutral-50 dark:bg-neutral-700/50 rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-danger-600 dark:text-danger-400">
              {faceStats.suspiciousFaces}
            </p>
            <p className="text-xs text-neutral-500 dark:text-neutral-400">Suspicious Faces</p>
          </div>
          <div className="bg-neutral-50 dark:bg-neutral-700/50 rounded-lg p-4 text-center">
            <p className={`text-2xl font-bold ${getScoreColor(faceStats.avgScore)}`}>
              {formatProbability(faceStats.avgScore)}
            </p>
            <p className="text-xs text-neutral-500 dark:text-neutral-400">Avg Score</p>
          </div>
        </div>
      )}

      {/* View mode toggle */}
      <div className="flex gap-2">
        <button
          onClick={() => setViewMode('grid')}
          className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
            viewMode === 'grid'
              ? 'bg-primary-100 text-primary-700 dark:bg-primary-900/30 dark:text-primary-300'
              : 'bg-neutral-100 text-neutral-600 dark:bg-neutral-700 dark:text-neutral-400 hover:bg-neutral-200 dark:hover:bg-neutral-600'
          }`}
        >
          Grid View
        </button>
        <button
          onClick={() => setViewMode('timeline')}
          className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
            viewMode === 'timeline'
              ? 'bg-primary-100 text-primary-700 dark:bg-primary-900/30 dark:text-primary-300'
              : 'bg-neutral-100 text-neutral-600 dark:bg-neutral-700 dark:text-neutral-400 hover:bg-neutral-200 dark:hover:bg-neutral-600'
          }`}
        >
          Timeline View
        </button>
      </div>

      {/* Face grid */}
      {viewMode === 'grid' && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
          {faces.map((face) => (
            <div
              key={face.faceId}
              onClick={() => setSelectedFaceId(selectedFaceId === face.faceId ? null : face.faceId)}
              className={`${getScoreBgColor(face.avgScore)} rounded-lg p-4 cursor-pointer transition-all hover:shadow-md ${
                selectedFaceId === face.faceId ? 'ring-2 ring-primary-500' : ''
              }`}
            >
              {/* Face icon placeholder */}
              <div className="aspect-square bg-neutral-200 dark:bg-neutral-600 rounded-lg mb-3 flex items-center justify-center">
                <User className="h-12 w-12 text-neutral-400 dark:text-neutral-500" />
              </div>

              {/* Face info */}
              <div className="text-center">
                <p className="text-sm font-medium text-neutral-800 dark:text-neutral-200">
                  Face #{face.faceId + 1}
                </p>
                <p className={`text-lg font-bold ${getScoreColor(face.avgScore)}`}>
                  {formatProbability(face.avgScore)}
                </p>
                <p className="text-xs text-neutral-500 dark:text-neutral-400">
                  {face.frameCount} appearances
                </p>
              </div>

              {/* Score indicator */}
              <div className="mt-2 flex items-center justify-center gap-1">
                {face.avgScore >= 0.5 ? (
                  <>
                    <AlertTriangle size={14} className="text-danger-500" />
                    <span className="text-xs text-danger-600 dark:text-danger-400">Suspicious</span>
                  </>
                ) : (
                  <>
                    <CheckCircle size={14} className="text-success-500" />
                    <span className="text-xs text-success-600 dark:text-success-400">Normal</span>
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Face timeline */}
      {viewMode === 'timeline' && (
        <div className="space-y-4">
          {faces.map((face) => (
            <div
              key={face.faceId}
              className="bg-white dark:bg-neutral-700/50 rounded-lg border border-neutral-200 dark:border-neutral-600 p-4"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-neutral-200 dark:bg-neutral-600 rounded-full flex items-center justify-center">
                    <User className="h-6 w-6 text-neutral-400" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-neutral-800 dark:text-neutral-200">
                      Face #{face.faceId + 1}
                    </p>
                    <p className="text-xs text-neutral-500 dark:text-neutral-400">
                      Detected in {face.frameCount} frames
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className={`text-lg font-bold ${getScoreColor(face.avgScore)}`}>
                    {formatProbability(face.avgScore)}
                  </p>
                  <p className="text-xs text-neutral-500 dark:text-neutral-400">Avg Score</p>
                </div>
              </div>

              {/* Score range */}
              <div className="flex items-center gap-4 mb-3 text-xs">
                <div className="flex items-center gap-1">
                  <TrendingDown size={14} className="text-success-500" />
                  <span className="text-neutral-600 dark:text-neutral-400">
                    Min: {formatProbability(face.minScore)}
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <TrendingUp size={14} className="text-danger-500" />
                  <span className="text-neutral-600 dark:text-neutral-400">
                    Max: {formatProbability(face.maxScore)}
                  </span>
                </div>
              </div>

              {/* Appearance timeline */}
              <div className="relative h-6 bg-neutral-100 dark:bg-neutral-800 rounded-full overflow-hidden">
                {face.appearances.map((app, idx) => {
                  const maxTimestamp = Math.max(...face.appearances.map((a) => a.timestamp));
                  const position = maxTimestamp > 0 ? (app.timestamp / maxTimestamp) * 100 : 0;

                  return (
                    <div
                      key={idx}
                      className={`absolute top-0 w-2 h-6 rounded-sm ${
                        app.score >= 0.7 ? 'bg-danger-500' :
                        app.score >= 0.4 ? 'bg-warning-500' : 'bg-success-500'
                      }`}
                      style={{ left: `calc(${position}% - 4px)` }}
                      title={`${formatTimestamp(app.timestamp)}: ${formatProbability(app.score)}`}
                    />
                  );
                })}
              </div>

              {/* Time labels */}
              <div className="flex justify-between mt-1 text-xs text-neutral-500 dark:text-neutral-400">
                <span>
                  {formatTimestamp(Math.min(...face.appearances.map((a) => a.timestamp)))}
                </span>
                <span>
                  {formatTimestamp(Math.max(...face.appearances.map((a) => a.timestamp)))}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Selected face details */}
      {selectedFaceId !== null && (
        <div className="bg-white dark:bg-neutral-700/50 rounded-lg border border-neutral-200 dark:border-neutral-600 p-6">
          <h3 className="text-sm font-semibold mb-4 text-neutral-800 dark:text-neutral-100">
            Face #{selectedFaceId + 1} Details
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-neutral-200 dark:border-neutral-600">
                  <th className="text-left py-2 px-3 text-neutral-500 dark:text-neutral-400">Time</th>
                  <th className="text-left py-2 px-3 text-neutral-500 dark:text-neutral-400">Score</th>
                  <th className="text-left py-2 px-3 text-neutral-500 dark:text-neutral-400">Detection Confidence</th>
                  <th className="text-left py-2 px-3 text-neutral-500 dark:text-neutral-400">Status</th>
                </tr>
              </thead>
              <tbody>
                {faces
                  .find((f) => f.faceId === selectedFaceId)
                  ?.appearances.map((app, idx) => (
                    <tr key={idx} className="border-b border-neutral-100 dark:border-neutral-700">
                      <td className="py-2 px-3 text-neutral-800 dark:text-neutral-200">
                        {formatTimestamp(app.timestamp)}
                      </td>
                      <td className={`py-2 px-3 font-medium ${getScoreColor(app.score)}`}>
                        {formatProbability(app.score)}
                      </td>
                      <td className="py-2 px-3 text-neutral-600 dark:text-neutral-400">
                        {formatProbability(app.confidence)}
                      </td>
                      <td className="py-2 px-3">
                        {app.score >= 0.5 ? (
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-danger-100 text-danger-700 dark:bg-danger-900/30 dark:text-danger-300">
                            Suspicious
                          </span>
                        ) : (
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-success-100 text-success-700 dark:bg-success-900/30 dark:text-success-300">
                            Normal
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default FaceAnalysisGrid;
