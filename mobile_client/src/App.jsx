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
import { C, RADIUS } from './tokens';

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
        <ActivityIndicator size="large" color={C.navy} />
      </SafeAreaView>
    );
  }

  if (!authenticated) {
    return (
      <SafeAreaView style={styles.root}>
        <StatusBar barStyle="light-content" backgroundColor={C.navy} />
        <LoginScreen onLoginSuccess={() => setAuthenticated(true)} />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar barStyle="light-content" backgroundColor={C.navy} />

      {/* App header — navy block */}
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
  root:   { flex: 1, backgroundColor: C.bg },
  center: { justifyContent: 'center', alignItems: 'center' },

  // Navy header block
  header: {
    paddingHorizontal: 20,
    paddingVertical:   14,
    backgroundColor:   C.navy,
    flexDirection:     'row',
    justifyContent:    'space-between',
    alignItems:        'center',
  },
  logo:     { color: C.textOnNavy, fontSize: 20, fontWeight: '700', letterSpacing: 0.5 },
  subtitle: { color: C.navySub,    fontSize: 12, marginTop: 2 },

  // Outlined white logout button
  logoutBtn: {
    paddingVertical:   6,
    paddingHorizontal: 12,
    borderRadius:      RADIUS.btn,
    borderWidth:       1,
    borderColor:       'rgba(255,255,255,0.6)',
  },
  logoutText: { color: C.textOnNavy, fontSize: 12, fontWeight: '600' },

  cameraWrapper: { flex: 1 },
});

export default App;
