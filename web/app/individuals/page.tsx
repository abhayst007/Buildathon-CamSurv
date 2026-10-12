'use client';

import React, { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import {
  Shield,
  Video,
  Users,
  Search,
  Filter,
  ArrowLeft,
  ArrowRight,
  Maximize2,
  Clock,
  Camera,
  Tag,
  Eye,
  RefreshCw,
  X,
  Play,
  Sparkles,
  ExternalLink,
  UserCheck,
  Database,
  SlidersHorizontal,
  ChevronRight,
  Layers,
} from 'lucide-react';

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

interface PersonFrame {
  id?: number;
  frame_path: string;
  timestamp: number;
  camera_id: string;
  crop_path?: string;
  confidence?: number;
  image_url?: string;
  crop_url?: string;
}

interface VideoSource {
  id: string;
  name: string;
  filename: string;
  category: string;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export default function IndividualsPage() {
  const [persons, setPersons] = useState<MappedPerson[]>([]);
  const [sources, setSources] = useState<VideoSource[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [stats, setStats] = useState<{ indexed_frames_count: number } | null>(null);

  // Filters & Search
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedCamera, setSelectedCamera] = useState<string>('all');
  const [faceFilter, setFaceFilter] = useState<'all' | 'faces_only' | 'profiles_only'>('all');
  const [sortBy, setSortBy] = useState<'sightings' | 'recent' | 'first_seen' | 'id'>('sightings');

  // Selected Person Dossier Modal
  const [selectedPerson, setSelectedPerson] = useState<MappedPerson | null>(null);
  const [personFrames, setPersonFrames] = useState<PersonFrame[]>([]);
  const [loadingFrames, setLoadingFrames] = useState<boolean>(false);
  const [inspectingFrame, setInspectingFrame] = useState<PersonFrame | null>(null);

  // Load initial data
  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    setLoading(true);
    try {
      const [personsRes, sourcesRes, healthRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/persons`),
        fetch(`${API_BASE_URL}/api/sources`),
        fetch(`${API_BASE_URL}/health`),
      ]);

      if (personsRes.ok) {
        const data = await personsRes.json();
        setPersons(data.persons || []);
      }
      if (sourcesRes.ok) {
        const data = await sourcesRes.json();
        setSources(data.sources || []);
      }
      if (healthRes.ok) {
        const data = await healthRes.json();
        setStats(data);
      }
    } catch {
      // Backend unreachable fallback
    } finally {
      setLoading(false);
    }
  };

  // Fetch sighting frames when an individual is opened
  const handleOpenDossier = async (person: MappedPerson) => {
    setSelectedPerson(person);
    setLoadingFrames(true);
    setPersonFrames([]);
    setInspectingFrame(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/persons/${person.person_id}/frames`);
      if (res.ok) {
        const data = await res.json();
        setPersonFrames(data.frames || []);
      }
    } catch {
      // Ignore
    } finally {
      setLoadingFrames(false);
    }
  };

  // Helper formatting
  const formatTimestamp = (seconds: number) => {
    if (isNaN(seconds)) return '00:00.00s';
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(2);
    return `${mins.toString().padStart(2, '0')}:${secs.padStart(5, '0')}s`;
  };

  const getAvatarUrl = (p: MappedPerson) => {
    if (p.avatar_url) {
      return p.avatar_url.startsWith('http') ? p.avatar_url : `${API_BASE_URL}${p.avatar_url}`;
    }
    return `${API_BASE_URL}/api/frame?path=${encodeURIComponent(p.avatar_path)}`;
  };

  const getCropUrl = (p: MappedPerson) => {
    if (p.crop_url) {
      return p.crop_url.startsWith('http') ? p.crop_url : `${API_BASE_URL}${p.crop_url}`;
    }
    if (p.full_crop_path) {
      return `${API_BASE_URL}/api/frame?path=${encodeURIComponent(p.full_crop_path)}`;
    }
    return null;
  };

  // Filtered & Sorted Persons
  const filteredPersons = useMemo(() => {
    return persons.filter((p) => {
      // Camera filter
      if (selectedCamera !== 'all' && p.camera_id !== selectedCamera) {
        return false;
      }
      // Face detection filter
      if (faceFilter === 'faces_only' && !p.has_face) {
        return false;
      }
      if (faceFilter === 'profiles_only' && p.has_face) {
        return false;
      }
      // Text query filter
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        const matchesId = p.person_id.toLowerCase().includes(q);
        const matchesDesc = p.visual_description.toLowerCase().includes(q);
        const matchesCam = p.camera_id.toLowerCase().includes(q);
        const matchesUpper = p.upper_color?.toLowerCase().includes(q);
        const matchesLower = p.lower_color?.toLowerCase().includes(q);
        if (!matchesId && !matchesDesc && !matchesCam && !matchesUpper && !matchesLower) {
          return false;
        }
      }
      return true;
    }).sort((a, b) => {
      if (sortBy === 'sightings') {
        return b.occurrence_count - a.occurrence_count;
      }
      if (sortBy === 'recent') {
        return b.last_seen_timestamp - a.last_seen_timestamp;
      }
      if (sortBy === 'first_seen') {
        return a.first_seen_timestamp - b.first_seen_timestamp;
      }
      return a.person_id.localeCompare(b.person_id);
    });
  }, [persons, selectedCamera, faceFilter, searchQuery, sortBy]);

  // Aggregate stats
  const totalFaces = useMemo(() => persons.filter((p) => p.has_face).length, [persons]);
  const totalSightings = useMemo(
    () => persons.reduce((acc, p) => acc + (p.occurrence_count || 1), 0),
    [persons]
  );
  const uniqueCameras = useMemo(
    () => new Set(persons.map((p) => p.camera_id)).size,
    [persons]
  );

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans">
      {/* Clean Header with Nav Links */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-40 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <Link href="/" className="flex items-center gap-3 group">
              <div className="p-2 bg-blue-50 border border-blue-200 rounded-lg text-blue-600 group-hover:bg-blue-100 transition-colors">
                <Shield className="w-5 h-5" />
              </div>
              <div>
                <span className="font-semibold text-lg text-slate-900 tracking-tight">
                  CamSurv
                </span>
                <p className="text-[11px] text-slate-500">Live Video Stream & Face Mapping System</p>
              </div>
            </Link>

            {/* Navigation Tabs */}
            <nav className="hidden md:flex items-center gap-1 pl-4 border-l border-slate-200">
              <Link
                href="/"
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors"
              >
                <Video className="w-4 h-4 text-slate-500" />
                <span>Live Feed & Search</span>
              </Link>
              <Link
                href="/individuals"
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-blue-50 text-blue-700 border border-blue-200"
              >
                <Users className="w-4 h-4 text-blue-600" />
                <span>Recorded Individuals</span>
                <span className="px-1.5 py-0.2 rounded-full bg-blue-100 text-blue-700 text-[10px] font-mono font-bold">
                  {persons.length}
                </span>
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
            <button
              onClick={fetchInitialData}
              disabled={loading}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-medium transition-colors cursor-pointer"
              title="Refresh directory"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
              <span className="hidden sm:inline">Refresh</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Mobile Navigation Bar */}
        <div className="flex md:hidden items-center justify-between bg-white p-2 rounded-xl border border-slate-200 text-xs">
          <Link
            href="/"
            className="flex-1 py-1.5 text-center text-slate-600 hover:text-slate-900 font-medium"
          >
            Live Feed & Search
          </Link>
          <div className="w-px h-4 bg-slate-200" />
          <Link
            href="/individuals"
            className="flex-1 py-1.5 text-center font-bold text-blue-600 bg-blue-50 rounded-lg"
          >
            Recorded Individuals ({persons.length})
          </Link>
        </div>

        {/* Directory Hero Banner & Telemetry KPIs */}
        <section className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
            <div>
              <div className="flex items-center gap-2">
                <div className="p-1.5 bg-blue-100 text-blue-700 rounded-lg">
                  <Users className="w-5 h-5" />
                </div>
                <h1 className="font-bold text-slate-900 text-base sm:text-lg">
                  Recorded Individuals & Face Directory
                </h1>
              </div>
              <p className="text-xs text-slate-500 mt-0.5">
                Catalog of all unique individuals identified across surveillance streams, mapped with facial features, appearance descriptions, and timeline occurrences.
              </p>
            </div>

            <Link
              href="/"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-300 hover:bg-slate-50 text-slate-700 text-xs font-medium self-start sm:self-auto transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Back to Live Stream</span>
            </Link>
          </div>

          {/* Quick Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
              <span className="text-[10px] text-slate-500 font-medium uppercase tracking-wider block">
                Total Unique Individuals
              </span>
              <span className="text-xl font-bold text-slate-900 font-mono mt-0.5 block">
                {persons.length}
              </span>
            </div>
            <div className="bg-blue-50/50 border border-blue-200 rounded-lg p-3">
              <span className="text-[10px] text-blue-700 font-medium uppercase tracking-wider block">
                Facial Crops Extracted
              </span>
              <span className="text-xl font-bold text-blue-700 font-mono mt-0.5 block">
                {totalFaces}
              </span>
            </div>
            <div className="bg-emerald-50/50 border border-emerald-200 rounded-lg p-3">
              <span className="text-[10px] text-emerald-700 font-medium uppercase tracking-wider block">
                Total Sightings Logged
              </span>
              <span className="text-xl font-bold text-emerald-700 font-mono mt-0.5 block">
                {totalSightings}
              </span>
            </div>
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
              <span className="text-[10px] text-slate-500 font-medium uppercase tracking-wider block">
                Surveillance Feeds
              </span>
              <span className="text-xl font-bold text-slate-900 font-mono mt-0.5 block">
                {uniqueCameras} Camera{uniqueCameras === 1 ? '' : 's'}
              </span>
            </div>
          </div>
        </section>

        {/* Filter and Search Bar */}
        <section className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-3 text-xs">
          <div className="flex flex-col md:flex-row md:items-center gap-3">
            {/* Search Input */}
            <div className="relative flex-1">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search individuals by ID (e.g. PER-001), clothing description, color, or camera..."
                className="w-full pl-9 pr-8 py-2 bg-slate-50 border border-slate-300 rounded-lg text-slate-900 text-xs focus:outline-none focus:border-blue-600 focus:bg-white transition-colors"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 cursor-pointer"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            {/* Camera Dropdown Filter */}
            <div className="flex items-center gap-1.5 shrink-0">
              <Camera className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-slate-500 font-medium">Camera:</span>
              <select
                value={selectedCamera}
                onChange={(e) => setSelectedCamera(e.target.value)}
                className="bg-slate-50 border border-slate-300 rounded-lg px-2.5 py-1.5 text-slate-800 text-xs focus:outline-none focus:border-blue-600 font-medium"
              >
                <option value="all">All Cameras ({persons.length})</option>
                {sources.map((src) => {
                  const count = persons.filter((p) => p.camera_id === src.id).length;
                  return (
                    <option key={src.id} value={src.id}>
                      {src.name} ({count})
                    </option>
                  );
                })}
              </select>
            </div>

            {/* Face Biomarker Filter */}
            <div className="flex items-center bg-slate-100 p-0.5 rounded-lg border border-slate-200 shrink-0">
              <button
                type="button"
                onClick={() => setFaceFilter('all')}
                className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors cursor-pointer ${
                  faceFilter === 'all'
                    ? 'bg-white text-slate-900 shadow-xs font-semibold'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                All ({persons.length})
              </button>
              <button
                type="button"
                onClick={() => setFaceFilter('faces_only')}
                className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors cursor-pointer ${
                  faceFilter === 'faces_only'
                    ? 'bg-white text-blue-600 shadow-xs font-semibold'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Faces ({totalFaces})
              </button>
              <button
                type="button"
                onClick={() => setFaceFilter('profiles_only')}
                className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors cursor-pointer ${
                  faceFilter === 'profiles_only'
                    ? 'bg-white text-slate-900 shadow-xs font-semibold'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Profiles ({persons.length - totalFaces})
              </button>
            </div>

            {/* Sort Dropdown */}
            <div className="flex items-center gap-1.5 shrink-0">
              <span className="text-slate-500 font-medium">Sort:</span>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="bg-slate-50 border border-slate-300 rounded-lg px-2.5 py-1.5 text-slate-800 text-xs focus:outline-none focus:border-blue-600 font-medium"
              >
                <option value="sightings">Most Sightings</option>
                <option value="recent">Recently Seen</option>
                <option value="first_seen">First Seen (Earliest)</option>
                <option value="id">Individual ID</option>
              </select>
            </div>
          </div>

          {/* Active Filter Indicators */}
          {(searchQuery || selectedCamera !== 'all' || faceFilter !== 'all') && (
            <div className="flex items-center gap-2 pt-2 border-t border-slate-100 text-[11px] text-slate-500">
              <span>Filtering: {filteredPersons.length} of {persons.length} individuals shown</span>
              <button
                onClick={() => {
                  setSearchQuery('');
                  setSelectedCamera('all');
                  setFaceFilter('all');
                }}
                className="text-blue-600 hover:underline font-medium cursor-pointer ml-auto"
              >
                Reset All Filters
              </button>
            </div>
          )}
        </section>

        {/* Individuals Directory Grid */}
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-slate-900 text-sm">
              Recorded Profiles ({filteredPersons.length} Individuals)
            </h2>
            <span className="text-xs text-slate-500">
              Click any individual card to open full details, visual description, and sighting frames
            </span>
          </div>

          {loading ? (
            <div className="bg-white border border-slate-200 rounded-xl p-16 text-center space-y-3">
              <RefreshCw className="w-8 h-8 text-blue-600 animate-spin mx-auto" />
              <p className="text-sm font-medium text-slate-700">Loading mapped individuals directory...</p>
            </div>
          ) : filteredPersons.length === 0 ? (
            <div className="bg-white border border-slate-200 rounded-xl p-16 text-center space-y-3">
              <Users className="w-10 h-10 text-slate-300 mx-auto" />
              <p className="text-sm font-medium text-slate-700">No individuals match your filter criteria</p>
              <p className="text-xs text-slate-500">
                Try clearing the search query or changing camera filter options.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {filteredPersons.map((person) => {
                const avatarSrc = getAvatarUrl(person);
                const cropSrc = getCropUrl(person);

                return (
                  <div
                    key={person.person_id}
                    onClick={() => handleOpenDossier(person)}
                    className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs hover:shadow-md hover:border-blue-400 transition-all flex flex-col cursor-pointer group"
                    title="Click to view full dossier, description, and sighting frames"
                  >
                    {/* Top Visual Header: Cropped Face + Body Preview */}
                    <div className="p-3.5 bg-slate-50 border-b border-slate-100 flex items-center justify-between gap-3">
                      <div className="flex items-center gap-3">
                        {/* Cropped Face Avatar */}
                        <div className="w-16 h-16 rounded-xl overflow-hidden bg-slate-200 shrink-0 border-2 border-blue-500/80 shadow-xs relative group-hover:scale-105 transition-transform">
                          <img
                            src={avatarSrc}
                            alt={person.person_id}
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              (e.target as HTMLElement).style.display = 'none';
                            }}
                          />
                          <span className="absolute bottom-0 inset-x-0 bg-blue-600/90 text-white text-[8px] font-mono text-center py-0.2">
                            {person.has_face ? 'Face' : 'Profile'}
                          </span>
                        </div>

                        {/* Full Body Crop (if present) */}
                        {cropSrc && (
                          <div className="w-12 h-16 rounded-lg overflow-hidden bg-slate-200 shrink-0 border border-slate-300 relative">
                            <img
                              src={cropSrc}
                              alt="Full body"
                              className="w-full h-full object-cover"
                            />
                            <span className="absolute bottom-0 inset-x-0 bg-black/70 text-white text-[7px] font-mono text-center py-0.2">
                              Body
                            </span>
                          </div>
                        )}
                      </div>

                      {/* Occurrences & Zoom Indicator */}
                      <div className="flex flex-col items-end gap-1.5">
                        <span className="px-2 py-0.5 rounded-full bg-blue-50 border border-blue-200 text-blue-700 text-[10px] font-mono font-bold">
                          {person.occurrence_count} sightings
                        </span>
                        <div className="p-1.5 rounded-lg bg-white border border-slate-200 text-slate-400 group-hover:text-blue-600 group-hover:border-blue-200 shadow-2xs transition-colors">
                          <Maximize2 className="w-3.5 h-3.5" />
                        </div>
                      </div>
                    </div>

                    {/* Body Details */}
                    <div className="p-3.5 flex-1 flex flex-col justify-between space-y-3">
                      <div className="space-y-1.5">
                        <div className="flex items-center justify-between">
                          <span className="font-bold text-slate-900 text-sm font-mono group-hover:text-blue-600 transition-colors">
                            {person.person_id}
                          </span>
                          <span className="text-[10px] font-mono text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded">
                            {person.camera_id}
                          </span>
                        </div>

                        <p className="text-xs text-slate-600 line-clamp-2 leading-relaxed">
                          {person.visual_description}
                        </p>
                      </div>

                      {/* Clothing Tag Pills */}
                      <div className="flex flex-wrap gap-1.5 pt-1">
                        {person.upper_color && (
                          <span className="px-2 py-0.5 bg-slate-100 text-slate-700 rounded text-[10px] font-medium capitalize border border-slate-200">
                            Upper: {person.upper_color}
                          </span>
                        )}
                        {person.lower_color && (
                          <span className="px-2 py-0.5 bg-slate-100 text-slate-700 rounded text-[10px] font-medium capitalize border border-slate-200">
                            Lower: {person.lower_color}
                          </span>
                        )}
                      </div>

                      {/* Footer Timestamps & CTA */}
                      <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-[10px] font-mono text-slate-400">
                        <span>First: {formatTimestamp(person.first_seen_timestamp)}</span>
                        <span>Last: {formatTimestamp(person.last_seen_timestamp)}</span>
                      </div>

                      <button
                        type="button"
                        className="w-full py-1.5 px-3 bg-slate-50 group-hover:bg-blue-50 group-hover:text-blue-700 group-hover:border-blue-200 border border-slate-200 text-slate-700 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 transition-colors"
                      >
                        <Eye className="w-3.5 h-3.5" />
                        <span>View Details & Sightings</span>
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </main>

      {/* Individual Dossier Lightbox Modal */}
      {selectedPerson && (
        <div
          className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-3 sm:p-6 animate-in fade-in duration-200 overflow-y-auto"
          onClick={() => setSelectedPerson(null)}
        >
          <div
            className="bg-white border border-slate-200 rounded-2xl max-w-4xl w-full overflow-hidden shadow-2xl flex flex-col max-h-[92vh] my-auto"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="p-4 border-b border-slate-100 flex items-center justify-between bg-slate-50">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-100 text-blue-700 rounded-lg">
                  <UserCheck className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-slate-900 text-base">
                    Individual Dossier: {selectedPerson.person_id}
                  </h3>
                  <p className="text-xs text-slate-500 font-mono">
                    Captured on feed: {selectedPerson.camera_id} • {selectedPerson.occurrence_count} recorded sightings
                  </p>
                </div>
              </div>

              <button
                onClick={() => setSelectedPerson(null)}
                className="p-1.5 text-slate-400 hover:text-slate-700 rounded-lg cursor-pointer hover:bg-slate-200/50 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Scrollable Body */}
            <div className="overflow-y-auto p-5 space-y-6 flex-1">
              {/* Profile Top Hero Section: Side-by-Side Zoomed Face & Body Crops */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-5 items-stretch bg-gradient-to-b from-slate-50 to-white p-4 rounded-xl border border-slate-200">
                {/* 1. Cropped Face Avatar */}
                <div className="flex flex-col items-center justify-center text-center space-y-2 p-2">
                  <div className="w-36 h-36 rounded-2xl overflow-hidden border-2 border-blue-500 shadow-md bg-black relative group">
                    <img
                      src={getAvatarUrl(selectedPerson)}
                      alt={selectedPerson.person_id}
                      className="w-full h-full object-cover"
                    />
                    <span className="absolute bottom-1 right-1 px-1.5 py-0.5 bg-blue-600 text-white text-[9px] font-mono rounded font-semibold shadow">
                      {selectedPerson.has_face ? 'Cropped Face' : 'Cropped Profile'}
                    </span>
                  </div>
                  <span className="text-xs font-semibold text-slate-800">High-Res Face Biomarker</span>
                  <span className="text-[10px] text-slate-500 font-mono">Unique ID: {selectedPerson.person_id}</span>
                </div>

                {/* 2. Full Subject Body Crop (if available) */}
                {getCropUrl(selectedPerson) ? (
                  <div className="flex flex-col items-center justify-center text-center space-y-2 p-2">
                    <div className="w-28 h-36 rounded-2xl overflow-hidden border border-slate-300 shadow-sm bg-black relative">
                      <img
                        src={getCropUrl(selectedPerson)!}
                        alt="Full subject crop"
                        className="w-full h-full object-cover"
                      />
                      <span className="absolute bottom-1 right-1 px-1.5 py-0.5 bg-black/80 text-white text-[8px] font-mono rounded font-semibold">
                        Full Body
                      </span>
                    </div>
                    <span className="text-xs font-semibold text-slate-800">Subject Body Crop</span>
                    <span className="text-[10px] text-slate-500 font-mono">Contextual Framing</span>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center text-center space-y-2 p-2 text-slate-400">
                    <Users className="w-12 h-12 stroke-1" />
                    <span className="text-xs">No full body crop logged</span>
                  </div>
                )}

                {/* 3. Appearance Metadata Card */}
                <div className="flex flex-col justify-between space-y-3 bg-white p-4 rounded-xl border border-slate-200">
                  <div className="space-y-1">
                    <span className="text-[10px] font-medium text-slate-400 uppercase tracking-wider block">
                      Visual Description
                    </span>
                    <p className="text-slate-800 text-xs font-medium leading-relaxed">
                      {selectedPerson.visual_description}
                    </p>
                  </div>

                  <div className="grid grid-cols-2 gap-2 pt-1 font-mono text-xs">
                    <div className="p-2 bg-slate-50 border border-slate-200 rounded-lg">
                      <span className="text-[9px] text-slate-400 block">UPPER WEAR</span>
                      <span className="font-semibold text-slate-800 capitalize">
                        {selectedPerson.upper_color || 'Unknown'}
                      </span>
                    </div>
                    <div className="p-2 bg-slate-50 border border-slate-200 rounded-lg">
                      <span className="text-[9px] text-slate-400 block">LOWER WEAR</span>
                      <span className="font-semibold text-slate-800 capitalize">
                        {selectedPerson.lower_color || 'Unknown'}
                      </span>
                    </div>
                    <div className="p-2 bg-slate-50 border border-slate-200 rounded-lg">
                      <span className="text-[9px] text-slate-400 block">FIRST SEEN</span>
                      <span className="font-semibold text-slate-800">
                        {formatTimestamp(selectedPerson.first_seen_timestamp)}
                      </span>
                    </div>
                    <div className="p-2 bg-slate-50 border border-slate-200 rounded-lg">
                      <span className="text-[9px] text-slate-400 block">LAST SEEN</span>
                      <span className="font-semibold text-slate-800">
                        {formatTimestamp(selectedPerson.last_seen_timestamp)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Sighting Timeline & Gallery */}
              <div className="space-y-3">
                <div className="flex items-center justify-between pb-2 border-b border-slate-100">
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-blue-600" />
                    <h4 className="font-semibold text-slate-900 text-sm">
                      Appearance Gallery & Sightings ({personFrames.length} Frames Recorded)
                    </h4>
                  </div>
                  <span className="text-xs text-slate-500">
                    Frames where {selectedPerson.person_id} was identified
                  </span>
                </div>

                {loadingFrames ? (
                  <div className="py-12 text-center text-xs text-slate-500 flex items-center justify-center gap-2">
                    <RefreshCw className="w-4 h-4 animate-spin text-blue-600" />
                    <span>Loading sighting frame records from database...</span>
                  </div>
                ) : personFrames.length === 0 ? (
                  <div className="py-8 text-center text-xs text-slate-400">
                    No additional frame records retrieved.
                  </div>
                ) : (
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                    {personFrames.map((f, idx) => {
                      const frameImgUrl = f.image_url
                        ? (f.image_url.startsWith('http') ? f.image_url : `${API_BASE_URL}${f.image_url}`)
                        : `${API_BASE_URL}/api/frame?path=${encodeURIComponent(f.frame_path)}`;
                      const frameCropUrl = f.crop_url
                        ? (f.crop_url.startsWith('http') ? f.crop_url : `${API_BASE_URL}${f.crop_url}`)
                        : null;

                      return (
                        <div
                          key={idx}
                          className="bg-white border border-slate-200 rounded-lg overflow-hidden shadow-2xs hover:shadow-md transition-shadow flex flex-col group relative"
                        >
                          {/* Sighting Image Preview */}
                          <div className="relative aspect-video bg-slate-900 overflow-hidden">
                            <img
                              src={frameImgUrl}
                              alt={`Sighting at ${f.timestamp}s`}
                              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                            />
                            {/* Timestamp tag */}
                            <div className="absolute top-1.5 left-1.5 px-1.5 py-0.5 bg-black/80 text-white rounded text-[9px] font-mono font-medium">
                              {formatTimestamp(f.timestamp)}
                            </div>
                            {/* Cropped Subject Inset */}
                            {frameCropUrl && (
                              <div className="absolute bottom-1 right-1 w-10 h-10 rounded overflow-hidden border border-white shadow bg-black">
                                <img
                                  src={frameCropUrl}
                                  alt="Crop"
                                  className="w-full h-full object-cover"
                                />
                              </div>
                            )}
                          </div>

                          {/* Sighting Footer Actions */}
                          <div className="p-2 space-y-1.5">
                            <div className="flex items-center justify-between text-[10px] font-mono text-slate-500">
                              <span>{f.camera_id}</span>
                              {f.confidence !== undefined && (
                                <span className="text-blue-600 font-semibold">
                                  {(f.confidence * 100).toFixed(0)}% conf
                                </span>
                              )}
                            </div>

                            <Link
                              href={`/?cam=${encodeURIComponent(f.camera_id)}&t=${f.timestamp}`}
                              className="w-full py-1 bg-slate-50 hover:bg-blue-50 hover:text-blue-700 hover:border-blue-200 border border-slate-200 text-slate-700 rounded text-[10px] font-medium flex items-center justify-center gap-1 transition-colors"
                              title="Jump to this moment in live video feed"
                            >
                              <Play className="w-3 h-3 fill-current" />
                              <span>Play in Feed</span>
                            </Link>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>

            {/* Modal Footer Controls */}
            <div className="p-4 bg-slate-50 border-t border-slate-100 flex items-center justify-between">
              <button
                onClick={() => setSelectedPerson(null)}
                className="px-4 py-2 border border-slate-300 rounded-lg text-slate-700 text-xs font-medium hover:bg-slate-100 cursor-pointer"
              >
                Close Dossier
              </button>

              <div className="flex items-center gap-2">
                <Link
                  href={`/?search=${encodeURIComponent(`Person ${selectedPerson.person_id}`)}`}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-medium flex items-center gap-1.5 shadow-xs transition-colors"
                >
                  <Search className="w-3.5 h-3.5" />
                  <span>Search Appearances in Video Footage</span>
                </Link>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
