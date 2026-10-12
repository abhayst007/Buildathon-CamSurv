'use client';

import React, { useState, useEffect, useRef, useMemo } from 'react';
import Link from 'next/link';
import {
  Search,
  Camera,
  Clock,
  Sparkles,
  SlidersHorizontal,
  Maximize2,
  X,
  Shield,
  Database,
  Cpu,
  RefreshCw,
  Eye,
  Layers,
  Plus,
  Play,
  Pause,
  User,
  Users,
  Video,
  ExternalLink,
  Activity,
  CheckCircle2,
  Filter,
  Maximize,
  Crosshair,
  Square,
  Scan,
  ArrowRight,
} from 'lucide-react';

interface ROIBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface ActivityTimespan {
  start_time: number;
  end_time: number;
  duration: number;
  camera_id: string;
  formatted_timespan: string;
  description: string;
  peak_similarity: number;
  frame_count: number;
  key_frame_url?: string;
  key_crop_url?: string;
  crop_url?: string;
  subject_class?: string;
}

interface SearchResult {
  frame_path: string;
  timestamp: number;
  camera_id: string;
  similarity_score: number;
  raw_similarity?: number;
  image_url?: string;
  crop_path?: string;
  crop_url?: string;
  crop_class?: string;
  crop_details?: Record<string, any>;
  in_roi?: boolean;
  metadata?: Record<string, any>;
}

interface SearchResponse {
  query: string;
  total_matches: number;
  results: SearchResult[];
  roi?: ROIBox;
  timespans?: ActivityTimespan[];
}

interface SystemStats {
  status: string;
  indexed_frames_count: number;
  clip_model: string;
  clip_device: string;
  vector_db: string;
}

interface VideoSource {
  id: string;
  name: string;
  category: string;
  filename: string;
  file_path: string;
  description: string;
  fps: number;
  duration: number;
  resolution: string;
  video_url: string;
  thumbnail_url?: string;
}

interface MappedPerson {
  id: number;
  person_id: string;
  camera_id: string;
  first_seen_timestamp: number;
  last_seen_timestamp: number;
  avatar_path: string;
  avatar_url?: string;
  full_crop_path?: string;
  crop_url?: string;
  frame_path: string;
  frame_url?: string;
  visual_description: string;
  upper_color?: string;
  lower_color?: string;
  dominant_color?: string;
  has_face: boolean;
  occurrence_count: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export default function SurveillanceDashboard() {
  // Search state
  const [query, setQuery] = useState('');
  const [selectedCamera, setSelectedCamera] = useState<string>('all');
  const [topK, setTopK] = useState<number>(8);
  const [minConfidence, setMinConfidence] = useState<number>(0.35);
  const [sortBy, setSortBy] = useState<'similarity' | 'time_asc' | 'time_desc'>('similarity');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [timespans, setTimespans] = useState<ActivityTimespan[]>([]);
  const [lastSearchedQuery, setLastSearchedQuery] = useState<string>('');
  const [hasSearched, setHasSearched] = useState(false);

  // Spatial Search Zone (ROI) state
  const [roiBox, setRoiBox] = useState<ROIBox | null>(null);
  const [isDrawingROI, setIsDrawingROI] = useState<boolean>(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [currentDrawBox, setCurrentDrawBox] = useState<ROIBox | null>(null);
  const videoContainerRef = useRef<HTMLDivElement | null>(null);

  // Modal inspection state
  const [selectedSnapshot, setSelectedSnapshot] = useState<SearchResult | null>(null);

  // Mapped face profile lightbox modal
  const [selectedFaceModal, setSelectedFaceModal] = useState<MappedPerson | null>(null);
  const [personFilterMode, setPersonFilterMode] = useState<'all' | 'active'>('all');

  // System telemetry & sources
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [sources, setSources] = useState<VideoSource[]>([]);
  const [activeLiveSource, setActiveLiveSource] = useState<VideoSource | null>(null);
  const [isSwitchingStream, setIsSwitchingStream] = useState<boolean>(false);

  // Mapped Persons state
  const [mappedPersons, setMappedPersons] = useState<MappedPerson[]>([]);
  const [selectedPersonFilter, setSelectedPersonFilter] = useState<string | null>(null);

  // Live Feed Player & Real-time Processing state
  const [isPlaying, setIsPlaying] = useState(true);
  const [liveCurrentTime, setLiveCurrentTime] = useState<number>(0);
  const [realtimeFramesProcessed, setRealtimeFramesProcessed] = useState<number>(0);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Filtered mapped persons by camera mode
  const displayedPersons = useMemo(() => {
    if (personFilterMode === 'active' && activeLiveSource) {
      return mappedPersons.filter((p) => p.camera_id === activeLiveSource.id);
    }
    return mappedPersons;
  }, [mappedPersons, personFilterMode, activeLiveSource]);

  // Ingestion modal state
  const [showIngestModal, setShowIngestModal] = useState(false);
  const [ingestPath, setIngestPath] = useState('');
  const [ingestCamId, setIngestCamId] = useState('');
  const [ingestFps, setIngestFps] = useState(1.0);
  const [ingestLoading, setIngestLoading] = useState(false);
  const [ingestMsg, setIngestMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Load initial data
  useEffect(() => {
    fetchSystemStats();
    fetchSources();
    fetchMappedPersons();
  }, []);

  // Update active source once sources load & handle incoming query parameters
  useEffect(() => {
    if (sources.length === 0) return;

    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      const camParam = params.get('cam') || params.get('camera');
      const timeParam = params.get('t') || params.get('time');
      const searchParam = params.get('search') || params.get('q');

      if (camParam) {
        const target = sources.find((s) => s.id === camParam);
        if (target) {
          setActiveLiveSource(target);
          setSelectedCamera(target.id);
        }
      } else if (!activeLiveSource) {
        setActiveLiveSource(sources[0]);
      }

      if (timeParam) {
        const t = parseFloat(timeParam);
        if (!isNaN(t)) {
          setTimeout(() => {
            if (videoRef.current) {
              videoRef.current.currentTime = t;
              videoRef.current.play().catch(() => {});
            }
          }, 600);
        }
      }

      if (searchParam) {
        setQuery(searchParam);
        handleSearch(undefined, searchParam);
      }
    } else if (!activeLiveSource) {
      setActiveLiveSource(sources[0]);
    }
  }, [sources]);

  // Handle stream switching with immediate prewarm & face mapping
  const handleSelectStream = async (targetId: string) => {
    const target = sources.find((s) => s.id === targetId);
    if (!target) return;
    setActiveLiveSource(target);
    setSelectedCamera(target.id);
    setIsSwitchingStream(true);

    try {
      const res = await fetch(`${API_BASE_URL}/api/realtime/switch-stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id: target.id, prewarm_samples: 5 }),
      });
      if (res.ok) {
        await fetchMappedPersons();
        await fetchSystemStats();
      }
    } catch {
      // Ignore
    } finally {
      setIsSwitchingStream(false);
    }
  };

  // Group sources by category for selector
  const categorizedSources = useMemo(() => {
    const map = new Map<string, VideoSource[]>();
    sources.forEach((src) => {
      const cat = src.category || 'General Feeds';
      if (!map.has(cat)) map.set(cat, []);
      map.get(cat)!.push(src);
    });
    return Array.from(map.entries());
  }, [sources]);

  // Real-time video frame snapshot and processing loop
  useEffect(() => {
    if (!isPlaying || !activeLiveSource) return;

    const interval = setInterval(async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.paused || video.ended) return;

      const currentTime = video.currentTime;
      setLiveCurrentTime(currentTime);

      let base64Data: string | undefined = undefined;
      try {
        canvas.width = Math.min(video.videoWidth || 640, 640);
        canvas.height = Math.min(video.videoHeight || 360, 360);
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          base64Data = canvas.toDataURL('image/jpeg', 0.7);
        }
      } catch {
        // Tainted canvas fallback - server will sample frame directly from video file
      }

      try {
        const payload: Record<string, any> = {
          camera_id: activeLiveSource.id,
          timestamp: currentTime,
        };
        if (base64Data) {
          payload.frame_base64 = base64Data;
        }

        const res = await fetch(`${API_BASE_URL}/api/realtime/process-frame`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (res.ok) {
          const data = await res.json();
          setRealtimeFramesProcessed((prev) => prev + 1);
          if (data.mapped_persons && data.mapped_persons.length > 0) {
            fetchMappedPersons();
          }
        }
      } catch {
        // Silent error handling for realtime capture
      }
    }, 2000); // Sample real-time processing every 2 seconds during playback

    return () => clearInterval(interval);
  }, [isPlaying, activeLiveSource]);

  const fetchSystemStats = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch {
      // Backend unreachable
    }
  };

  const fetchSources = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/sources`);
      if (res.ok) {
        const data = await res.json();
        if (data.sources) {
          setSources(data.sources);
        }
      }
    } catch {
      // Fallback default
    }
  };

  const fetchMappedPersons = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/persons`);
      if (res.ok) {
        const data = await res.json();
        if (data.persons) {
          setMappedPersons(data.persons);
        }
      }
    } catch {
      // Fallback
    }
  };

  const handleSearch = async (e?: React.FormEvent, overrideQuery?: string) => {
    if (e) e.preventDefault();
    const searchQuery = (overrideQuery ?? query).trim();
    if (!searchQuery) return;

    setLoading(true);
    setError(null);
    setHasSearched(true);
    setLastSearchedQuery(searchQuery);
    setSelectedPersonFilter(null);

    try {
      const payload: Record<string, any> = {
        query: searchQuery,
        top_k: topK,
      };

      if (selectedCamera !== 'all') {
        payload.camera_id = selectedCamera;
      }
      if (minConfidence > 0) {
        payload.min_similarity = minConfidence;
      }
      if (roiBox) {
        payload.roi = roiBox;
      }

      const res = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${res.status}`);
      }

      const data: SearchResponse = await res.json();
      setResults(data.results || []);
      setTimespans(data.timespans || []);
    } catch (err: any) {
      setError(err.message || 'Failed to complete search.');
    } finally {
      setLoading(false);
    }
  };

  const handleRoiMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDrawingROI || !videoContainerRef.current) return;
    const rect = videoContainerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    setDrawStart({ x, y });
    setCurrentDrawBox({ x1: x, y1: y, x2: x, y2: y });
  };

  const handleRoiMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDrawingROI || !drawStart || !videoContainerRef.current) return;
    const rect = videoContainerRef.current.getBoundingClientRect();
    const currentX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const currentY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

    const x1 = Math.min(drawStart.x, currentX);
    const x2 = Math.max(drawStart.x, currentX);
    const y1 = Math.min(drawStart.y, currentY);
    const y2 = Math.max(drawStart.y, currentY);

    setCurrentDrawBox({ x1, y1, x2, y2 });
  };

  const handleRoiMouseUp = () => {
    if (!isDrawingROI || !drawStart || !currentDrawBox) {
      setDrawStart(null);
      setCurrentDrawBox(null);
      return;
    }
    const width = currentDrawBox.x2 - currentDrawBox.x1;
    const height = currentDrawBox.y2 - currentDrawBox.y1;
    if (width > 0.02 && height > 0.02) {
      setRoiBox({
        x1: Math.round(currentDrawBox.x1 * 1000) / 1000,
        y1: Math.round(currentDrawBox.y1 * 1000) / 1000,
        x2: Math.round(currentDrawBox.x2 * 1000) / 1000,
        y2: Math.round(currentDrawBox.y2 * 1000) / 1000,
      });
    }
    setDrawStart(null);
    setCurrentDrawBox(null);
    setIsDrawingROI(false);
  };

  const jumpToTimespan = (ts: ActivityTimespan) => {
    const targetSource = sources.find((s) => s.id === ts.camera_id);
    if (targetSource && targetSource.id !== activeLiveSource?.id) {
      setActiveLiveSource(targetSource);
      setSelectedCamera(targetSource.id);
    }
    const video = videoRef.current;
    if (video) {
      video.currentTime = ts.start_time;
      video.play().catch(() => {});
      setIsPlaying(true);
    }
  };

  const filterByPerson = async (person: MappedPerson) => {
    setSelectedPersonFilter(person.person_id);
    setLoading(true);
    setError(null);
    setHasSearched(true);
    setLastSearchedQuery(`Person ${person.person_id}`);

    try {
      const res = await fetch(`${API_BASE_URL}/api/persons/${person.person_id}/frames`);
      if (res.ok) {
        const data = await res.json();
        const mappedFrames: SearchResult[] = (data.frames || []).map((f: any) => ({
          frame_path: f.frame_path,
          timestamp: f.timestamp,
          camera_id: f.camera_id,
          similarity_score: 0.95,
          raw_similarity: f.confidence || 0.95,
          image_url: `/api/frame?path=${encodeURIComponent(f.frame_path)}`,
          metadata: {
            matched_detection: {
              class: 'Person',
              matched_color: person.upper_color,
              person_id: person.person_id,
            },
          },
        }));
        setResults(mappedFrames);
      }
    } catch (err: any) {
      setError('Could not retrieve frames for person.');
    } finally {
      setLoading(false);
    }
  };

  const handleIngestVideo = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!ingestPath.trim()) return;

    setIngestLoading(true);
    setIngestMsg(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: ingestPath.trim(),
          camera_id: ingestCamId.trim() || undefined,
          target_fps: ingestFps,
          run_yolo: true,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Ingestion failed (${res.status})`);
      }

      const data = await res.json();
      setIngestMsg({
        type: 'success',
        text: `Success! Extracted and indexed ${data.frames_extracted} frames.`,
      });
      fetchSystemStats();
      fetchSources();
      fetchMappedPersons();
      setTimeout(() => {
        setShowIngestModal(false);
        setIngestMsg(null);
        setIngestPath('');
        setIngestCamId('');
      }, 2000);
    } catch (err: any) {
      setIngestMsg({ type: 'error', text: err.message || 'Error ingesting video.' });
    } finally {
      setIngestLoading(false);
    }
  };

  // Processed and sorted results
  const processedResults = useMemo(() => {
    const list = [...results];
    if (sortBy === 'time_asc') {
      list.sort((a, b) => a.timestamp - b.timestamp);
    } else if (sortBy === 'time_desc') {
      list.sort((a, b) => b.timestamp - a.timestamp);
    } else {
      list.sort((a, b) => b.similarity_score - a.similarity_score);
    }
    return list;
  }, [results, sortBy]);

  const formatTimestamp = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(2);
    return `${mins.toString().padStart(2, '0')}:${secs.padStart(5, '0')}s`;
  };

  const getImageUrl = (item: SearchResult) => {
    if (item.image_url) {
      return item.image_url.startsWith('http')
        ? item.image_url
        : `${API_BASE_URL}${item.image_url}`;
    }
    return `${API_BASE_URL}/api/frame?path=${encodeURIComponent(item.frame_path)}`;
  };

  const getScoreBadge = (score: number) => {
    const percent = Math.round(score * 100);
    if (percent >= 75) {
      return {
        text: `${percent}% Confidence`,
        classes: 'bg-emerald-50 text-emerald-700 border-emerald-200',
      };
    }
    if (percent >= 50) {
      return {
        text: `${percent}% Match`,
        classes: 'bg-blue-50 text-blue-700 border-blue-200',
      };
    }
    return {
      text: `${percent}% Low`,
      classes: 'bg-amber-50 text-amber-700 border-amber-200',
    };
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans">
      {/* Hidden canvas for real-time video snapshot processing */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Clean Minimalist Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-40 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <Link href="/" className="flex items-center gap-3 group">
              <div className="p-2 bg-blue-50 border border-blue-200 rounded-lg text-blue-600 group-hover:bg-blue-100 transition-colors">
                <Shield className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-lg text-slate-900 tracking-tight">
                    CamSurv
                  </span>
                </div>
                <p className="text-xs text-slate-500">Live Video Stream & Face Mapping System</p>
              </div>
            </Link>

            {/* Top Navigation Tabs */}
            <nav className="hidden md:flex items-center gap-1 pl-4 border-l border-slate-200">
              <Link
                href="/"
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-blue-50 text-blue-700 border border-blue-200"
              >
                <Video className="w-4 h-4 text-blue-600" />
                <span>Live Feed & Search</span>
              </Link>
              <Link
                href="/individuals"
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors"
                title="View directory of all recorded individuals"
              >
                <Users className="w-4 h-4 text-slate-500" />
                <span>Recorded Individuals</span>
                {mappedPersons.length > 0 && (
                  <span className="px-1.5 py-0.2 rounded-full bg-blue-100 text-blue-700 text-[10px] font-mono font-bold">
                    {mappedPersons.length}
                  </span>
                )}
              </Link>
            </nav>
          </div>

          <div className="flex items-center gap-3">
            {stats && (
              <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 border border-slate-200 text-slate-600 text-xs font-mono">
                <Database className="w-3.5 h-3.5 text-blue-600" />
                <span>{stats.indexed_frames_count} Frames Indexed</span>
              </div>
            )}
            {mappedPersons.length > 0 && (
              <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 border border-slate-200 text-slate-600 text-xs font-mono">
                <Users className="w-3.5 h-3.5 text-blue-600" />
                <span>{mappedPersons.length} Individuals Mapped</span>
              </div>
            )}
            <button
              onClick={() => setShowIngestModal(true)}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium transition-colors shadow-sm cursor-pointer"
            >
              <Plus className="w-4 h-4" />
              <span>Add Source</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Mobile Navigation Tabs */}
        <div className="flex md:hidden items-center justify-between bg-white p-2 rounded-xl border border-slate-200 text-xs shadow-2xs">
          <Link
            href="/"
            className="flex-1 py-1.5 text-center font-bold text-blue-600 bg-blue-50 rounded-lg flex items-center justify-center gap-1.5"
          >
            <Video className="w-3.5 h-3.5" />
            <span>Live Stream</span>
          </Link>
          <div className="w-px h-4 bg-slate-200 mx-1" />
          <Link
            href="/individuals"
            className="flex-1 py-1.5 text-center text-slate-600 hover:text-slate-900 font-medium flex items-center justify-center gap-1.5"
          >
            <Users className="w-3.5 h-3.5" />
            <span>Individuals ({mappedPersons.length})</span>
          </Link>
        </div>

        {/* 1. Live Camera Feed Section */}
        <section className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
            <div className="flex items-center gap-2">
              <Video className="w-5 h-5 text-blue-600" />
              <div>
                <h2 className="font-semibold text-slate-900 text-sm sm:text-base">
                  Live Surveillance Feed
                </h2>
                <p className="text-xs text-slate-500">
                  Parallel real-time visual analysis, face mapping, and indexing
                </p>
              </div>
            </div>

            {/* Categorized Camera Selector */}
            <div className="flex items-center gap-2 text-xs">
              <label htmlFor="feed-selector" className="text-slate-500 font-medium">Select Feed:</label>
              <select
                id="feed-selector"
                value={activeLiveSource?.id || ''}
                onChange={(e) => handleSelectStream(e.target.value)}
                className="bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-slate-800 text-xs focus:outline-none focus:border-blue-600 font-medium"
              >
                {categorizedSources.map(([category, items]) => (
                  <optgroup key={category} label={category}>
                    {items.map((src) => (
                      <option key={src.id} value={src.id}>
                        {src.name} ({src.filename})
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </div>
          </div>

          {/* Live Video Player and Telemetry */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 items-start">
            {/* HTML5 Video Player with Interactive Spatial Drawing Layer */}
            <div
              ref={videoContainerRef}
              onMouseDown={handleRoiMouseDown}
              onMouseMove={handleRoiMouseMove}
              onMouseUp={handleRoiMouseUp}
              onMouseLeave={handleRoiMouseUp}
              className={`lg:col-span-2 relative bg-slate-900 rounded-lg overflow-hidden aspect-video flex items-center justify-center shadow-inner ${
                isDrawingROI ? 'cursor-crosshair select-none ring-2 ring-emerald-500' : ''
              }`}
            >
              {activeLiveSource ? (
                <>
                  <video
                    ref={videoRef}
                    key={activeLiveSource.id}
                    crossOrigin="anonymous"
                    src={`${API_BASE_URL}/api/video/${activeLiveSource.id}`}
                    controls={!isDrawingROI}
                    autoPlay
                    loop
                    muted
                    playsInline
                    onLoadedMetadata={(e) => {
                      e.currentTarget.play().catch(() => {});
                    }}
                    onPlay={() => setIsPlaying(true)}
                    onPause={() => setIsPlaying(false)}
                    onTimeUpdate={(e) => setLiveCurrentTime(e.currentTarget.currentTime)}
                    className="w-full h-full object-contain"
                  />
                  {isSwitchingStream && (
                    <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center text-white text-xs gap-2 z-20 pointer-events-none">
                      <RefreshCw className="w-4 h-4 animate-spin text-blue-400" />
                      <span>Analyzing stream & mapping faces...</span>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-slate-400 text-xs flex items-center gap-2">
                  <Activity className="w-4 h-4 animate-spin text-blue-400" />
                  <span>Loading surveillance feed...</span>
                </div>
              )}

              {/* Feed Name Tag */}
              {activeLiveSource && (
                <div className="absolute top-3 left-3 px-2.5 py-1 bg-black/70 backdrop-blur-sm text-white rounded text-[11px] font-mono flex items-center gap-1.5 pointer-events-none z-10">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span>{activeLiveSource.name}</span>
                </div>
              )}

              {/* Drawing Toolbar on Top-Right */}
              <div className="absolute top-3 right-3 flex items-center gap-2 z-20">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setIsDrawingROI(!isDrawingROI);
                  }}
                  className={`px-2.5 py-1 rounded text-xs font-medium flex items-center gap-1.5 shadow-md backdrop-blur-sm transition-all cursor-pointer ${
                    isDrawingROI
                      ? 'bg-emerald-600 text-white ring-2 ring-white/50'
                      : roiBox
                      ? 'bg-black/75 hover:bg-black/90 text-emerald-400 border border-emerald-500/50'
                      : 'bg-black/75 hover:bg-black/90 text-white border border-white/20'
                  }`}
                  title="Draw a rectangular search area/square on the video"
                >
                  <Crosshair className="w-3.5 h-3.5" />
                  <span>{isDrawingROI ? 'Drawing Zone...' : roiBox ? 'Redraw Zone' : 'Draw Search Zone'}</span>
                </button>

                {roiBox && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setRoiBox(null);
                    }}
                    className="px-2 py-1 rounded bg-black/75 hover:bg-rose-600 text-white text-xs font-medium flex items-center gap-1 shadow-md backdrop-blur-sm transition-all cursor-pointer border border-white/20"
                    title="Clear Drawn Zone"
                  >
                    <X className="w-3.5 h-3.5" />
                    <span>Clear</span>
                  </button>
                )}
              </div>

              {/* Drawing Guide Prompt Banner */}
              {isDrawingROI && (
                <div className="absolute bottom-3 inset-x-3 bg-black/85 backdrop-blur-md text-emerald-300 text-xs py-2 px-3 rounded-lg border border-emerald-500/50 flex items-center justify-between z-20 pointer-events-none">
                  <div className="flex items-center gap-2">
                    <Scan className="w-4 h-4 text-emerald-400 animate-pulse" />
                    <span>Click and drag anywhere on the video frame to trace a search zone square</span>
                  </div>
                  <span className="text-[10px] text-white/70 font-mono">Drag to trace • Release to finish</span>
                </div>
              )}

              {/* In-Progress Drawn Box (while dragging) */}
              {isDrawingROI && currentDrawBox && (
                <div
                  className="absolute border-2 border-dashed border-emerald-400 bg-emerald-500/25 pointer-events-none z-20 rounded"
                  style={{
                    left: `${currentDrawBox.x1 * 100}%`,
                    top: `${currentDrawBox.y1 * 100}%`,
                    width: `${(currentDrawBox.x2 - currentDrawBox.x1) * 100}%`,
                    height: `${(currentDrawBox.y2 - currentDrawBox.y1) * 100}%`,
                  }}
                >
                  <div className="absolute top-1 left-1 bg-emerald-600 text-white text-[9px] font-mono px-1 py-0.2 rounded shadow">
                    Tracing Area...
                  </div>
                </div>
              )}

              {/* Committed ROI Box (active search zone) */}
              {roiBox && !isDrawingROI && (
                <div
                  className="absolute border-2 border-emerald-400 bg-emerald-500/20 pointer-events-none z-10 rounded shadow-lg ring-1 ring-emerald-300/50"
                  style={{
                    left: `${roiBox.x1 * 100}%`,
                    top: `${roiBox.y1 * 100}%`,
                    width: `${(roiBox.x2 - roiBox.x1) * 100}%`,
                    height: `${(roiBox.y2 - roiBox.y1) * 100}%`,
                  }}
                >
                  <div className="absolute -top-6 left-0 bg-emerald-600 text-white text-[10px] font-mono px-2 py-0.5 rounded shadow flex items-center gap-1.5 whitespace-nowrap">
                    <Crosshair className="w-3 h-3" />
                    <span>Search Zone ({(roiBox.x1 * 100).toFixed(0)}%, {(roiBox.y1 * 100).toFixed(0)}%)</span>
                  </div>
                </div>
              )}
            </div>

            {/* Sidebar: Available Video Source Cards & Live Telemetry */}
            <div className="space-y-3">
              {/* 1. Available Video Sources Cards Deck */}
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5 space-y-2.5 text-xs">
                <div className="flex items-center justify-between pb-1.5 border-b border-slate-200">
                  <div className="flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-blue-600" />
                    <span className="font-semibold text-slate-900 text-xs">
                      Available Video Feeds ({sources.length})
                    </span>
                  </div>
                  <span className="text-[10px] text-slate-500 font-mono">
                    Click card to feed live
                  </span>
                </div>

                {/* Vertical Scrollable Deck of Feed Cards */}
                <div className="space-y-2 max-h-[350px] overflow-y-auto pr-0.5">
                  {sources.map((src) => {
                    const isActive = activeLiveSource?.id === src.id;
                    const thumbUrl = src.thumbnail_url
                      ? (src.thumbnail_url.startsWith('http') ? src.thumbnail_url : `${API_BASE_URL}${src.thumbnail_url}`)
                      : `${API_BASE_URL}/api/sources/${src.id}/thumbnail`;

                    return (
                      <div
                        key={src.id}
                        onClick={() => handleSelectStream(src.id)}
                        className={`p-2 rounded-lg border transition-all cursor-pointer flex gap-2.5 items-center group relative ${
                          isActive
                            ? 'bg-blue-50/90 border-blue-600 shadow-xs ring-1 ring-blue-500/50'
                            : 'bg-white border-slate-200 hover:border-blue-300 hover:bg-slate-50/80 shadow-2xs'
                        }`}
                        title={`Click to feed ${src.name} into live surveillance player`}
                      >
                        {/* Single frame preview thumbnail from this video source */}
                        <div className="w-22 h-14 rounded-md overflow-hidden bg-slate-900 shrink-0 border border-slate-300 relative group-hover:scale-102 transition-transform shadow-2xs">
                          <img
                            src={thumbUrl}
                            alt={src.name}
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              (e.target as HTMLElement).style.display = 'none';
                            }}
                          />
                          {isActive ? (
                            <div className="absolute top-1 left-1 px-1.5 py-0.5 bg-emerald-600 text-white text-[8px] font-mono font-bold rounded flex items-center gap-1 shadow-xs">
                              <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                              <span>LIVE</span>
                            </div>
                          ) : (
                            <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                              <Play className="w-3.5 h-3.5 text-white fill-current drop-shadow" />
                            </div>
                          )}
                          <span className="absolute bottom-0.5 right-1 text-[8px] font-mono text-white/90 bg-black/75 px-1 rounded">
                            {src.duration ? `${src.duration}s` : 'Feed'}
                          </span>
                        </div>

                        {/* Card metadata */}
                        <div className="min-w-0 flex-1 space-y-0.5">
                          <div className="flex items-center justify-between">
                            <span
                              className={`text-xs font-bold truncate ${
                                isActive ? 'text-blue-700' : 'text-slate-900 group-hover:text-blue-600'
                              }`}
                            >
                              {src.name}
                            </span>
                            {isActive && (
                              <span className="text-[9px] font-mono bg-blue-100 text-blue-700 font-bold px-1.5 py-0.2 rounded shrink-0">
                                ACTIVE
                              </span>
                            )}
                          </div>

                          <p className="text-[10px] text-slate-500 truncate">
                            {src.category}
                          </p>

                          <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 pt-0.5">
                            <span>{src.resolution}</span>
                            <span className={isActive ? 'text-emerald-700 font-semibold' : 'text-slate-500'}>
                              {isActive ? 'Streaming Now' : 'Click to Feed →'}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* 2. Live Feed Parallel Processing Telemetry */}
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5 space-y-3 text-xs">
                <div className="space-y-1">
                  <span className="text-[10px] font-medium text-slate-400 uppercase tracking-wider block">
                    Processing Mode
                  </span>
                  <div className="flex items-center gap-2 text-emerald-700 bg-emerald-50 px-2.5 py-1.5 rounded-md border border-emerald-200 font-medium">
                    <Activity className="w-3.5 h-3.5 animate-spin" />
                    <span>Parallel Live Stream Ingestion Active</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2.5 font-mono">
                  <div className="p-2 bg-white border border-slate-200 rounded-md">
                    <span className="text-[9px] text-slate-400 block">TIME OFFSET</span>
                    <span className="text-slate-800 font-bold text-xs sm:text-sm">
                      {formatTimestamp(liveCurrentTime)}
                    </span>
                  </div>
                  <div className="p-2 bg-white border border-slate-200 rounded-md">
                    <span className="text-[9px] text-slate-400 block">SAMPLED FRAMES</span>
                    <span className="text-blue-600 font-bold text-xs sm:text-sm">
                      {realtimeFramesProcessed}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. Mapped Unique Individuals Section */}
        <section className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-slate-100">
            <div className="flex items-center gap-2">
              <Users className="w-4 h-4 text-blue-600" />
              <h2 className="font-semibold text-slate-900 text-sm">
                Mapped Individuals ({displayedPersons.length} Persons Displayed)
              </h2>
            </div>

            <div className="flex items-center gap-2">
              {/* Feed Filter Toggle */}
              <div className="flex items-center bg-slate-100 p-0.5 rounded-lg text-[11px] font-medium border border-slate-200">
                <button
                  type="button"
                  onClick={() => setPersonFilterMode('all')}
                  className={`px-2.5 py-1 rounded-md transition-colors cursor-pointer ${
                    personFilterMode === 'all'
                      ? 'bg-white text-slate-900 shadow-xs font-semibold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  All Feeds ({mappedPersons.length})
                </button>
                {activeLiveSource && (
                  <button
                    type="button"
                    onClick={() => setPersonFilterMode('active')}
                    className={`px-2.5 py-1 rounded-md transition-colors cursor-pointer ${
                      personFilterMode === 'active'
                        ? 'bg-white text-blue-600 shadow-xs font-semibold'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Active Feed ({mappedPersons.filter((p) => p.camera_id === activeLiveSource.id).length})
                  </button>
                )}
              </div>

              <Link
                href="/individuals"
                className="text-xs text-blue-600 hover:text-blue-800 font-medium flex items-center gap-1 transition-colors px-2 py-1 rounded hover:bg-blue-50 border border-transparent hover:border-blue-200"
                title="Open dedicated individuals & face directory"
              >
                <span>Full Directory</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>

              {selectedPersonFilter && (
                <button
                  onClick={() => {
                    setSelectedPersonFilter(null);
                    setResults([]);
                    setHasSearched(false);
                  }}
                  className="text-xs text-blue-600 hover:text-blue-700 flex items-center gap-1 font-medium cursor-pointer ml-1"
                >
                  <span>Clear Filter ({selectedPersonFilter})</span>
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>

          <div className="overflow-x-auto pb-2">
            <div className="flex items-stretch gap-3 min-w-max">
              {displayedPersons.length === 0 ? (
                <div className="py-6 px-4 text-center text-xs text-slate-400 font-mono w-full flex items-center justify-center gap-2">
                  <Activity className="w-4 h-4 text-slate-300" />
                  <span>No individuals mapped for this feed yet. Real-time stream analysis is running.</span>
                </div>
              ) : (
                displayedPersons.map((person) => {
                  const isSelected = selectedPersonFilter === person.person_id;
                  const avatarSrc = person.avatar_url
                    ? `${API_BASE_URL}${person.avatar_url}`
                    : `${API_BASE_URL}/api/frame?path=${encodeURIComponent(person.avatar_path)}`;

                  return (
                    <div
                      key={person.person_id}
                      onClick={() => setSelectedFaceModal(person)}
                      className={`p-3 rounded-lg border transition-all cursor-pointer flex gap-3 w-76 items-center group relative hover:shadow-md ${
                        isSelected
                          ? 'border-blue-600 bg-blue-50/50 shadow-sm'
                          : 'border-slate-200 bg-slate-50 hover:border-blue-300 hover:bg-white'
                      }`}
                      title="Click to pop up enlarged cropped face and details"
                    >
                      {/* Person Avatar Thumbnail (Click to zoom) */}
                      <div className="w-16 h-16 rounded-lg overflow-hidden bg-slate-200 shrink-0 border border-slate-300 relative group-hover:border-blue-500 transition-colors shadow-xs">
                        <img
                          src={avatarSrc}
                          alt={person.person_id}
                          className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-200"
                          onError={(e) => {
                            (e.target as HTMLElement).style.display = 'none';
                          }}
                        />
                        <span className="absolute bottom-0 right-0 bg-blue-600 text-white px-1 py-0.2 rounded-tl text-[8px] font-mono">
                          {person.has_face ? 'Face' : 'Profile'}
                        </span>
                        <div className="absolute inset-0 bg-blue-600/10 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                          <Maximize className="w-3.5 h-3.5 text-blue-700 bg-white/80 rounded p-0.5 shadow-xs" />
                        </div>
                      </div>

                      {/* Person Details */}
                      <div className="min-w-0 flex-1 space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="font-bold text-slate-900 text-xs font-mono group-hover:text-blue-600 transition-colors">
                            {person.person_id}
                          </span>
                          <span className="text-[10px] px-1.5 py-0.5 bg-white border border-slate-200 rounded text-slate-600 font-mono">
                            {person.occurrence_count} sightings
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-600 line-clamp-2 leading-tight">
                          {person.visual_description}
                        </p>
                        <div className="text-[10px] text-slate-400 font-mono flex items-center justify-between pt-0.5">
                          <span className="truncate max-w-[100px]">{person.camera_id}</span>
                          <span>{formatTimestamp(person.first_seen_timestamp)}</span>
                        </div>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </section>

        {/* 3. Search Console & Controls */}
        <section className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4">
          {/* Spatial ROI Zone Active Banner */}
          {roiBox && (
            <div className="flex items-center justify-between bg-emerald-50 border border-emerald-200 rounded-lg px-3.5 py-2.5 text-xs animate-in fade-in">
              <div className="flex items-center gap-2 text-emerald-800">
                <Crosshair className="w-4 h-4 text-emerald-600 shrink-0" />
                <span>
                  <strong className="font-semibold">Traced Spatial Zone Active:</strong> X: [{(roiBox.x1 * 100).toFixed(0)}%–{(roiBox.x2 * 100).toFixed(0)}%], Y: [{(roiBox.y1 * 100).toFixed(0)}%–{(roiBox.y2 * 100).toFixed(0)}%]
                </span>
                <span className="text-[11px] text-emerald-600 hidden sm:inline font-normal">
                  — searching specifically within this traced area and calculating crossing timespans
                </span>
              </div>
              <button
                type="button"
                onClick={() => setRoiBox(null)}
                className="text-emerald-700 hover:text-emerald-900 flex items-center gap-1 text-[11px] font-medium cursor-pointer ml-2 hover:underline"
              >
                <span>Clear Zone</span>
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

          <form onSubmit={handleSearch} className="flex gap-2">
            <div className="relative flex-1">
              <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search surveillance footage (e.g. person wearing red t-shirt, dark hoodie, counter, street)..."
                className="w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-300 rounded-lg text-slate-900 text-sm focus:outline-none focus:border-blue-600 focus:bg-white transition-colors"
              />
            </div>
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors shadow-sm flex items-center gap-2 cursor-pointer"
            >
              {loading && <RefreshCw className="w-4 h-4 animate-spin" />}
              <span>Search</span>
            </button>
          </form>

          {/* Filter Bar */}
          <div className="flex flex-wrap items-center justify-between gap-4 pt-2 border-t border-slate-100 text-xs">
            <div className="flex flex-wrap items-center gap-4">
              {/* Categorized Camera Filter */}
              <div className="flex items-center gap-1.5">
                <Camera className="w-4 h-4 text-slate-400" />
                <span className="text-slate-500 font-medium">Filter Camera:</span>
                <select
                  value={selectedCamera}
                  onChange={(e) => setSelectedCamera(e.target.value)}
                  className="bg-slate-50 border border-slate-300 rounded-md px-2.5 py-1 text-slate-800 text-xs focus:outline-none focus:border-blue-600"
                >
                  <option value="all">All Cameras ({sources.length})</option>
                  {categorizedSources.map(([category, items]) => (
                    <optgroup key={category} label={category}>
                      {items.map((src) => (
                        <option key={src.id} value={src.id}>
                          {src.name}
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </select>
              </div>

              {/* Max Results */}
              <div className="flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-slate-400" />
                <span className="text-slate-500 font-medium">Max Results:</span>
                <select
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="bg-slate-50 border border-slate-300 rounded-md px-2.5 py-1 text-slate-800 text-xs focus:outline-none focus:border-blue-600"
                >
                  {[4, 8, 12, 16, 24].map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </div>

              {/* Confidence Slider */}
              <div className="flex items-center gap-2">
                <SlidersHorizontal className="w-4 h-4 text-slate-400" />
                <span className="text-slate-500 font-medium">Min Confidence:</span>
                <input
                  type="range"
                  min="0.10"
                  max="0.90"
                  step="0.05"
                  value={minConfidence}
                  onChange={(e) => setMinConfidence(Number(e.target.value))}
                  className="w-20 accent-blue-600 cursor-pointer"
                />
                <span className="font-mono text-blue-600 font-semibold w-8">
                  {Math.round(minConfidence * 100)}%
                </span>
              </div>
            </div>

            {/* Sort Order */}
            <div className="flex items-center gap-1.5">
              <span className="text-slate-500 font-medium">Sort:</span>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="bg-slate-50 border border-slate-300 rounded-md px-2.5 py-1 text-slate-800 text-xs focus:outline-none focus:border-blue-600"
              >
                <option value="similarity">Confidence (Highest First)</option>
                <option value="time_asc">Timestamp (Earliest)</option>
                <option value="time_desc">Timestamp (Latest)</option>
              </select>
            </div>
          </div>
        </section>

        {/* 4. Search Results Grid */}
        {hasSearched && (
          <section className="space-y-4">
            {/* Activity Timespans Banner (Timespans when subject passed through traced area) */}
            {timespans.length > 0 && (
              <div className="bg-emerald-50/70 border border-emerald-200 rounded-xl p-4 space-y-3 shadow-xs animate-in fade-in">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 bg-emerald-100 text-emerald-800 rounded-lg shrink-0">
                      <Clock className="w-4 h-4 text-emerald-700" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-slate-900 text-sm flex items-center gap-2">
                        <span>Activity Timespans in Traced Search Zone</span>
                        <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-emerald-200/80 text-emerald-800 font-bold">
                          {timespans.length} Interval{timespans.length === 1 ? '' : 's'}
                        </span>
                      </h4>
                      <p className="text-xs text-slate-600">
                        Contiguous time intervals when matching subjects passed through the traced square/area.
                      </p>
                    </div>
                  </div>
                  <span className="text-[10px] font-mono bg-emerald-100 text-emerald-800 px-2.5 py-1 rounded font-semibold border border-emerald-200 self-start sm:self-auto">
                    SPATIAL ZONE MATCH
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 pt-1">
                  {timespans.map((ts, idx) => {
                    const keyframeSrc = ts.key_frame_url
                      ? (ts.key_frame_url.startsWith('http') ? ts.key_frame_url : `${API_BASE_URL}${ts.key_frame_url}`)
                      : null;
                    const cropRaw = ts.key_crop_url || ts.crop_url;
                    const cropSrc = cropRaw
                      ? (cropRaw.startsWith('http') ? cropRaw : `${API_BASE_URL}${cropRaw}`)
                      : null;

                    return (
                      <div
                        key={idx}
                        className="bg-white border border-emerald-200 rounded-lg p-3 shadow-xs hover:shadow-md transition-shadow flex flex-col justify-between space-y-2.5 group"
                      >
                        <div className="flex gap-3 items-center">
                          {keyframeSrc && (
                            <div className="w-20 h-14 rounded-md overflow-hidden bg-slate-900 relative shrink-0 border border-slate-200 shadow-xs">
                              <img src={keyframeSrc} alt="Keyframe" className="w-full h-full object-cover" />
                              {cropSrc && (
                                <div className="absolute bottom-0.5 right-0.5 w-6 h-6 rounded overflow-hidden border border-white shadow bg-black/80">
                                  <img src={cropSrc} alt="Crop" className="w-full h-full object-cover" />
                                </div>
                              )}
                            </div>
                          )}

                          <div className="min-w-0 flex-1 space-y-0.5">
                            <div className="flex items-center justify-between">
                              <span className="font-mono font-bold text-xs text-emerald-800">
                                {ts.formatted_timespan}
                              </span>
                              <span className="text-[10px] font-mono text-slate-500 bg-slate-100 px-1.5 py-0.2 rounded">
                                {ts.duration}s span
                              </span>
                            </div>
                            <p className="text-[11px] text-slate-700 truncate font-medium">
                              {ts.description}
                            </p>
                            <div className="flex items-center justify-between text-[10px] font-mono text-slate-500">
                              <span>{ts.camera_id}</span>
                              <span className="text-emerald-700 font-semibold">{Math.round(ts.peak_similarity * 100)}% match</span>
                            </div>
                          </div>
                        </div>

                        <button
                          type="button"
                          onClick={() => jumpToTimespan(ts)}
                          className="w-full py-1.5 px-3 bg-emerald-600 hover:bg-emerald-700 text-white rounded-md text-xs font-medium flex items-center justify-center gap-1.5 transition-colors shadow-xs cursor-pointer"
                        >
                          <Play className="w-3.5 h-3.5 fill-current" />
                          <span>Jump & Play in Feed ({formatTimestamp(ts.start_time)})</span>
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-slate-900 text-sm">
                Search Results: {processedResults.length} match{processedResults.length === 1 ? '' : 'es'}
                {lastSearchedQuery && (
                  <span className="text-slate-500 font-normal ml-1">
                    for "{lastSearchedQuery}"
                  </span>
                )}
              </h3>
            </div>

            {processedResults.length === 0 ? (
              <div className="bg-white border border-slate-200 rounded-xl p-12 text-center space-y-2">
                <Search className="w-8 h-8 text-slate-300 mx-auto" />
                <p className="font-medium text-slate-700 text-sm">No matching frames found</p>
                <p className="text-xs text-slate-500">
                  Try lowering the confidence threshold or broadening the search description.
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {processedResults.map((item, idx) => {
                  const badge = getScoreBadge(item.similarity_score);
                  const cropImageUrl = item.crop_url
                    ? (item.crop_url.startsWith('http') ? item.crop_url : `${API_BASE_URL}${item.crop_url}`)
                    : null;

                  return (
                    <div
                      key={`${item.frame_path}-${idx}`}
                      className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-shadow flex flex-col group"
                    >
                      {/* Image snapshot: Whole Frame with Cropped Target Inset */}
                      <div className="relative aspect-video bg-slate-900 overflow-hidden">
                        {/* 1. Whole Frame */}
                        <img
                          src={getImageUrl(item)}
                          alt={item.camera_id}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                        />
                        <div className="absolute top-2 left-2 px-2 py-0.5 bg-black/75 backdrop-blur-sm text-white rounded text-[10px] font-mono font-medium flex items-center gap-1 z-10">
                          <Camera className="w-3 h-3 text-blue-400" />
                          <span>{item.camera_id}</span>
                        </div>
                        {item.in_roi && (
                          <div className="absolute top-8 left-2 px-2 py-0.5 bg-emerald-600/90 backdrop-blur-sm text-white rounded text-[10px] font-mono font-medium flex items-center gap-1 z-10 shadow">
                            <Crosshair className="w-3 h-3" />
                            <span>In Traced Zone</span>
                          </div>
                        )}
                        <div
                          className={`absolute top-2 right-2 px-2 py-0.5 rounded text-[10px] font-medium border z-10 ${badge.classes}`}
                        >
                          {badge.text}
                        </div>

                        {/* 2. Cropped Property Inset (User requirement: both whole frame & cropped property visible) */}
                        {cropImageUrl && (
                          <div
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedSnapshot(item);
                            }}
                            className="absolute bottom-2 right-2 w-16 h-16 rounded-lg overflow-hidden border-2 border-white shadow-xl bg-black/80 backdrop-blur-sm z-10 flex flex-col cursor-pointer hover:scale-105 transition-transform"
                            title="Cropped Target Subject (Click to inspect)"
                          >
                            <img
                              src={cropImageUrl}
                              alt="Cropped target property"
                              className="w-full h-full object-cover"
                            />
                            <span className="absolute bottom-0 inset-x-0 bg-black/85 text-white text-[8px] font-mono font-bold text-center py-0.2 capitalize truncate">
                              {item.crop_class || 'Crop'}
                            </span>
                          </div>
                        )}
                      </div>

                      {/* Card Body */}
                      <div className="p-3 flex-1 flex flex-col justify-between space-y-2.5">
                        <div className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-slate-500 flex items-center gap-1">
                              <Clock className="w-3.5 h-3.5 text-slate-400" />
                              Timestamp
                            </span>
                            <span className="font-mono text-slate-800 font-medium">
                              {formatTimestamp(item.timestamp)}
                            </span>
                          </div>

                          {/* Matched / Cropped Detection Info */}
                          {item.crop_details ? (
                            <div className="flex items-center gap-1.5 text-[10px] bg-blue-50 border border-blue-200 text-blue-700 px-2 py-0.5 rounded font-medium">
                              <Sparkles className="w-3 h-3 text-blue-600 shrink-0" />
                              <span className="truncate">
                                {item.crop_details.upper_color ? `${item.crop_details.upper_color.toUpperCase()} ` : ''}
                                {item.crop_class?.toUpperCase() || 'SUBJECT'} DETECTED
                              </span>
                            </div>
                          ) : item.metadata?.matched_detection && (
                            <div className="flex items-center gap-1.5 text-[10px] bg-blue-50 border border-blue-200 text-blue-700 px-2 py-0.5 rounded font-medium">
                              <Sparkles className="w-3 h-3 text-blue-600 shrink-0" />
                              <span className="truncate">
                                {item.metadata.matched_detection.matched_color
                                  ? `${item.metadata.matched_detection.matched_color.toUpperCase()} `
                                  : ''}
                                {item.metadata.matched_detection.class?.toUpperCase()} DETECTED
                              </span>
                            </div>
                          )}

                          <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 pt-0.5">
                            <span>{item.timestamp.toFixed(2)}s</span>
                            {item.raw_similarity !== undefined && (
                              <span>Raw: {item.raw_similarity.toFixed(3)}</span>
                            )}
                          </div>
                        </div>

                        <button
                          onClick={() => setSelectedSnapshot(item)}
                          className="w-full py-1.5 bg-slate-50 hover:bg-blue-50 hover:border-blue-200 hover:text-blue-700 border border-slate-200 text-slate-700 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
                        >
                          <Eye className="w-3.5 h-3.5 text-slate-500" />
                          <span>Inspect Frame & Crop</span>
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        )}
      </main>

      {/* Snapshot Inspection Modal: Dual Whole Frame & Cropped Property View */}
      {selectedSnapshot && (
        <div
          className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 animate-in fade-in duration-200"
          onClick={() => setSelectedSnapshot(null)}
        >
          <div
            className="bg-white border border-slate-200 rounded-2xl max-w-4xl w-full overflow-hidden shadow-2xl flex flex-col max-h-[90vh]"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="p-4 border-b border-slate-100 flex items-center justify-between bg-slate-50">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-slate-900 text-sm">
                  {selectedSnapshot.camera_id} Inspection
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  {formatTimestamp(selectedSnapshot.timestamp)}
                </span>
              </div>
              <button
                onClick={() => setSelectedSnapshot(null)}
                className="p-1 text-slate-400 hover:text-slate-700 rounded-lg cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Dual Frame & Crop Display */}
            <div className="p-4 bg-slate-900 grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
              {/* 1. Whole Video Frame (Scene Context) */}
              <div
                className={`${
                  selectedSnapshot.crop_url ? 'md:col-span-2' : 'md:col-span-3'
                } relative rounded-lg overflow-hidden bg-black flex flex-col items-center justify-center border border-slate-700`}
              >
                <div className="absolute top-2 left-2 z-10 px-2 py-0.5 bg-black/75 text-white text-[10px] font-mono rounded">
                  Whole Frame (Scene Context)
                </div>
                <img
                  src={getImageUrl(selectedSnapshot)}
                  alt="Full frame"
                  className="max-h-[50vh] w-full object-contain"
                />
              </div>

              {/* 2. Cropped Property Target View */}
              {selectedSnapshot.crop_url && (
                <div className="rounded-lg overflow-hidden bg-slate-800/90 border border-slate-700 p-3 flex flex-col items-center justify-center space-y-2">
                  <div className="w-full flex items-center justify-between text-white text-xs pb-1 border-b border-slate-700">
                    <span className="font-semibold">Cropped Property</span>
                    <span className="font-mono text-[10px] uppercase text-emerald-400">
                      {selectedSnapshot.crop_class || 'Target'}
                    </span>
                  </div>
                  <div className="w-full h-44 rounded-md overflow-hidden bg-black flex items-center justify-center border border-slate-600">
                    <img
                      src={
                        selectedSnapshot.crop_url.startsWith('http')
                          ? selectedSnapshot.crop_url
                          : `${API_BASE_URL}${selectedSnapshot.crop_url}`
                      }
                      alt="Cropped property"
                      className="w-full h-full object-contain"
                    />
                  </div>
                  {selectedSnapshot.crop_details && (
                    <div className="w-full text-[11px] text-slate-300 space-y-1 font-mono pt-1">
                      {selectedSnapshot.crop_details.upper_color && (
                        <div className="flex justify-between">
                          <span className="text-slate-400">Upper Wear:</span>
                          <span className="capitalize">{selectedSnapshot.crop_details.upper_color}</span>
                        </div>
                      )}
                      {selectedSnapshot.crop_details.lower_color && (
                        <div className="flex justify-between">
                          <span className="text-slate-400">Lower Wear:</span>
                          <span className="capitalize">{selectedSnapshot.crop_details.lower_color}</span>
                        </div>
                      )}
                      {selectedSnapshot.crop_details.confidence !== undefined && (
                        <div className="flex justify-between">
                          <span className="text-slate-400">YOLO Conf:</span>
                          <span>{(selectedSnapshot.crop_details.confidence * 100).toFixed(1)}%</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Modal Telemetry Bar */}
            <div className="p-4 bg-slate-50 border-t border-slate-100 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-2.5 bg-white border border-slate-200 rounded-lg">
                <span className="text-[10px] text-slate-400 block font-medium">MATCH CONFIDENCE</span>
                <span className="font-bold text-emerald-600 text-sm">
                  {(selectedSnapshot.similarity_score * 100).toFixed(1)}%
                </span>
              </div>
              <div className="p-2.5 bg-white border border-slate-200 rounded-lg">
                <span className="text-[10px] text-slate-400 block font-medium">RAW COSINE SIM</span>
                <span className="font-bold text-slate-800 text-sm">
                  {selectedSnapshot.raw_similarity !== undefined
                    ? selectedSnapshot.raw_similarity.toFixed(4)
                    : selectedSnapshot.similarity_score.toFixed(4)}
                </span>
              </div>
              <div className="p-2.5 bg-white border border-slate-200 rounded-lg">
                <span className="text-[10px] text-slate-400 block font-medium">TIMESTAMP</span>
                <span className="font-bold text-slate-800 text-sm">
                  {selectedSnapshot.timestamp.toFixed(2)}s
                </span>
              </div>
              <div className="p-2.5 bg-white border border-slate-200 rounded-lg">
                <span className="text-[10px] text-slate-400 block font-medium">CAMERA</span>
                <span className="font-bold text-slate-800 text-sm">
                  {selectedSnapshot.camera_id}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Face Crop Popup Modal (User requirement: when clicked on face, cropped face pops up) */}
      {selectedFaceModal && (
        <div
          className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 animate-in fade-in duration-200"
          onClick={() => setSelectedFaceModal(null)}
        >
          <div
            className="bg-white border border-slate-200 rounded-2xl max-w-lg w-full overflow-hidden shadow-2xl space-y-0"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="p-4 border-b border-slate-100 flex items-center justify-between bg-slate-50">
              <div className="flex items-center gap-2">
                <div className="p-1.5 bg-blue-100 text-blue-700 rounded-md">
                  <User className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="font-semibold text-slate-900 text-sm">
                    Target Individual Profile: {selectedFaceModal.person_id}
                  </h3>
                  <p className="text-[11px] text-slate-500 font-mono">
                    Captured on camera: {selectedFaceModal.camera_id}
                  </p>
                </div>
              </div>
              <button
                onClick={() => setSelectedFaceModal(null)}
                className="p-1 text-slate-400 hover:text-slate-700 rounded-lg cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Cropped Face & Full Body Display */}
            <div className="p-5 flex flex-col sm:flex-row items-center justify-center gap-5 bg-gradient-to-b from-white to-slate-50">
              {/* Cropped Face Zoom */}
              <div className="flex flex-col items-center gap-1.5">
                <div className="w-32 h-32 rounded-xl overflow-hidden border-2 border-blue-500 shadow-md bg-slate-100 shrink-0 relative group">
                  <img
                    src={
                      selectedFaceModal.avatar_url
                        ? `${API_BASE_URL}${selectedFaceModal.avatar_url}`
                        : `${API_BASE_URL}/api/frame?path=${encodeURIComponent(selectedFaceModal.avatar_path)}`
                    }
                    alt={selectedFaceModal.person_id}
                    className="w-full h-full object-cover"
                  />
                  <span className="absolute bottom-1 right-1 px-1.5 py-0.5 bg-blue-600/90 text-white text-[9px] font-mono rounded">
                    {selectedFaceModal.has_face ? 'Face Crop' : 'Profile Crop'}
                  </span>
                </div>
                <span className="text-[10px] text-slate-500 font-medium">Cropped Face Avatar</span>
              </div>

              {/* Full Body Crop (if available) */}
              {selectedFaceModal.full_crop_path && (
                <div className="flex flex-col items-center gap-1.5">
                  <div className="w-24 h-32 rounded-xl overflow-hidden border border-slate-200 shadow-sm bg-slate-100 shrink-0 relative">
                    <img
                      src={
                        selectedFaceModal.crop_url
                          ? `${API_BASE_URL}${selectedFaceModal.crop_url}`
                          : `${API_BASE_URL}/api/frame?path=${encodeURIComponent(selectedFaceModal.full_crop_path)}`
                      }
                      alt="Full body crop"
                      className="w-full h-full object-cover"
                    />
                    <span className="absolute bottom-1 right-1 px-1 py-0.5 bg-black/70 text-white text-[8px] font-mono rounded">
                      Body
                    </span>
                  </div>
                  <span className="text-[10px] text-slate-500 font-medium">Full Subject Crop</span>
                </div>
              )}
            </div>

            {/* Profile Metadata */}
            <div className="p-5 border-t border-slate-100 space-y-3 text-xs">
              <div>
                <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">
                  Visual Appearance Description
                </span>
                <p className="text-slate-800 font-medium text-sm mt-0.5">
                  {selectedFaceModal.visual_description}
                </p>
              </div>

              <div className="grid grid-cols-3 gap-2.5 pt-1">
                <div className="p-2 bg-slate-50 border border-slate-200 rounded-lg">
                  <span className="text-[10px] text-slate-400 block">UPPER WEAR</span>
                  <span className="font-semibold text-slate-800 capitalize">
                    {selectedFaceModal.upper_color || 'Unknown'}
                  </span>
                </div>
                <div className="p-2 bg-slate-50 border border-slate-200 rounded-lg">
                  <span className="text-[10px] text-slate-400 block">LOWER WEAR</span>
                  <span className="font-semibold text-slate-800 capitalize">
                    {selectedFaceModal.lower_color || 'Unknown'}
                  </span>
                </div>
                <div className="p-2 bg-slate-50 border border-slate-200 rounded-lg">
                  <span className="text-[10px] text-slate-400 block">SIGHTINGS</span>
                  <span className="font-semibold text-blue-600 font-mono">
                    {selectedFaceModal.occurrence_count} occurrences
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between text-slate-500 text-[11px] pt-1 font-mono">
                <span>First Seen: {formatTimestamp(selectedFaceModal.first_seen_timestamp)}</span>
                <span>Last Seen: {formatTimestamp(selectedFaceModal.last_seen_timestamp)}</span>
              </div>
            </div>

            {/* Modal Actions */}
            <div className="p-4 bg-slate-50 border-t border-slate-100 flex items-center justify-between">
              <button
                onClick={() => setSelectedFaceModal(null)}
                className="px-4 py-2 border border-slate-300 rounded-lg text-slate-700 text-xs font-medium hover:bg-slate-100 cursor-pointer"
              >
                Close
              </button>
              <button
                onClick={() => {
                  const p = selectedFaceModal;
                  setSelectedFaceModal(null);
                  filterByPerson(p);
                }}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-medium flex items-center gap-1.5 shadow-sm cursor-pointer"
              >
                <Search className="w-3.5 h-3.5" />
                <span>Search All Appearances</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Ingest Modal */}
      {showIngestModal && (
        <div
          className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4"
          onClick={() => setShowIngestModal(false)}
        >
          <div
            className="bg-white border border-slate-200 rounded-2xl max-w-md w-full overflow-hidden shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-4 border-b border-slate-100 flex items-center justify-between">
              <h3 className="font-semibold text-slate-900 text-sm">Add New Surveillance Video</h3>
              <button
                onClick={() => setShowIngestModal(false)}
                className="p-1 text-slate-400 hover:text-slate-700 rounded-lg"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleIngestVideo} className="p-5 space-y-4 text-xs">
              <div>
                <label className="block text-slate-600 font-medium mb-1">
                  Video File Path
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. sources/my_video.mp4"
                  value={ingestPath}
                  onChange={(e) => setIngestPath(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-slate-900 focus:outline-none focus:border-blue-600"
                />
              </div>

              <div>
                <label className="block text-slate-600 font-medium mb-1">
                  Camera Identifier (Optional)
                </label>
                <input
                  type="text"
                  placeholder="e.g. camera_entrance_02"
                  value={ingestCamId}
                  onChange={(e) => setIngestCamId(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-slate-900 focus:outline-none focus:border-blue-600"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-600 font-medium mb-1">
                  <span>Frame Sampling Rate</span>
                  <span className="text-blue-600 font-bold">{ingestFps} FPS</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="5.0"
                  step="0.5"
                  value={ingestFps}
                  onChange={(e) => setIngestFps(parseFloat(e.target.value))}
                  className="w-full accent-blue-600"
                />
              </div>

              {ingestMsg && (
                <div
                  className={`p-3 rounded-lg text-xs ${
                    ingestMsg.type === 'success'
                      ? 'bg-emerald-50 border border-emerald-200 text-emerald-700'
                      : 'bg-rose-50 border border-rose-200 text-rose-700'
                  }`}
                >
                  {ingestMsg.text}
                </div>
              )}

              <div className="pt-2 flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowIngestModal(false)}
                  disabled={ingestLoading}
                  className="px-3.5 py-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={ingestLoading || !ingestPath.trim()}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium transition-colors shadow-sm cursor-pointer"
                >
                  {ingestLoading && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
                  <span>{ingestLoading ? 'Processing...' : 'Start Ingest'}</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
