/**
 * App.jsx
 * -------
 * Root component of the Vision OMR mobile client.
 *
 * State machine:
 *   idle ──► scanning ──► uploading ──► results ──► idle
 *                └──────────────────────────────────────┘
 *                          (re-scan)
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  SafeAreaView,
  View,
  Text,
  StyleSheet,
  StatusBar,
  Alert,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';

import CameraScanner from './components/CameraScanner';
import ResultsModal  from './components/ResultsModal';
import LoginScreen   from './components/LoginScreen';
import {
  evaluateSheet,
  submitResults,
  getStoredToken,
  logoutUser,
  setOnUnauthorized,
} from './services/api';

// ── Screens ────────────────────────────────────────────────────────────────

const SCREEN = {
  IDLE:      'idle',
  SCANNING:  'scanning',
  RESULTS:   'results',
};

// ── Component ──────────────────────────────────────────────────────────────

const App = () => {
  const [authenticated, setAuthenticated] = useState(false);
  const [authChecking, setAuthChecking]   = useState(true);
  const [screen,  setScreen]              = useState(SCREEN.SCANNING);
  const [results, setResults]             = useState(null);
  const [loading, setLoading]             = useState(false);

  useEffect(() => {
    setOnUnauthorized(() => {
      setAuthenticated(false);
      setResults(null);
      setScreen(SCREEN.SCANNING);
    });

    getStoredToken().then(token => {
      setAuthenticated(!!token);
      setAuthChecking(false);
    });
  }, []);

  const handleLogout = useCallback(async () => {
    await logoutUser();
    setAuthenticated(false);
    setResults(null);
    setScreen(SCREEN.SCANNING);
  }, []);

  // ── Handlers ─────────────────────────────────────────────────────────────

  const handleCapture = useCallback(async (photo) => {
    setLoading(true);
    setScreen(SCREEN.RESULTS); // show modal with spinner immediately

    try {
      const data = await evaluateSheet(photo);
      setResults(data);
    } catch (err) {
      Alert.alert('Evaluation Error', err.message);
      setScreen(SCREEN.SCANNING);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!results) return;
    try {
      await submitResults(results);
      Alert.alert('Submitted', 'Results saved successfully.', [
        { text: 'OK', onPress: () => { setResults(null); setScreen(SCREEN.SCANNING); } },
      ]);
    } catch (err) {
      Alert.alert('Submit Error', err.message);
    }
  }, [results]);

  const handleRescan = useCallback(() => {
    setResults(null);
    setScreen(SCREEN.SCANNING);
  }, []);

  const handleCloseModal = useCallback(() => {
    if (!loading) handleRescan();
  }, [loading, handleRescan]);

  // ── Render ────────────────────────────────────────────────────────────────

  if (authChecking) {
    return (
      <SafeAreaView style={[styles.root, styles.center]}>
        <ActivityIndicator size="large" color="#6366f1" />
      </SafeAreaView>
    );
  }

  if (!authenticated) {
    return (
      <SafeAreaView style={styles.root}>
        <StatusBar barStyle="light-content" backgroundColor="#0a0a14" />
        <LoginScreen onLoginSuccess={() => setAuthenticated(true)} />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar barStyle="light-content" backgroundColor="#0a0a14" />

      {/* App header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.logo}>Vision OMR</Text>
          <Text style={styles.subtitle}>Point camera at answer sheet</Text>
        </View>
        <TouchableOpacity style={styles.logoutBtn} onPress={handleLogout}>
          <Text style={styles.logoutText}>Logout</Text>
        </TouchableOpacity>
      </View>

      {/* Camera view */}
      <View style={styles.cameraWrapper}>
        <CameraScanner
          onCapture={handleCapture}
          onError={err => Alert.alert('Camera Error', err.message)}
        />
      </View>

      {/* Results modal */}
      <ResultsModal
        visible={screen === SCREEN.RESULTS}
        results={results}
        loading={loading}
        onSubmit={handleSubmit}
        onRescan={handleRescan}
        onClose={handleCloseModal}
      />
    </SafeAreaView>
  );
};

// ── Styles ─────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  root:          { flex: 1, backgroundColor: '#0a0a14' },
  center:        { justifyContent: 'center', alignItems: 'center' },

  header: {
    paddingHorizontal: 20,
    paddingVertical:   14,
    borderBottomWidth: 1,
    borderBottomColor: '#1e1e2e',
    flexDirection:     'row',
    justifyContent:    'space-between',
    alignItems:        'center',
  },
  logo:          { color: '#e2e8f0', fontSize: 20, fontWeight: '800', letterSpacing: 0.5 },
  subtitle:      { color: '#64748b', fontSize: 12, marginTop: 2 },
  logoutBtn:     { paddingVertical: 6, paddingHorizontal: 12, borderRadius: 6, backgroundColor: '#1e1e2e' },
  logoutText:    { color: '#ef4444', fontSize: 12, fontWeight: '600' },

  cameraWrapper: { flex: 1 },
});

export default App;
