import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { loginUser } from '../services/api';
import { C, RADIUS } from '../tokens';

const LoginScreen = ({ onLoginSuccess }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!username.trim() || !password) {
      Alert.alert('Validation Error', 'Please enter username and password.');
      return;
    }

    setLoading(true);
    try {
      await loginUser(username.trim(), password);
      onLoginSuccess();
    } catch (err) {
      Alert.alert('Login Failed', err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.root}>
      {/* ── Navy header block ─────────────────────────────────────────── */}
      <View style={styles.headerBlock}>
        <View style={styles.iconBadge}>
          <Text style={styles.iconBadgeText}>📋</Text>
        </View>
        <Text style={styles.appName}>Vision OMR</Text>
        <Text style={styles.appSub}>Sign in to continue</Text>
      </View>

      {/* ── White card body ───────────────────────────────────────────── */}
      <View style={styles.card}>
        {/* Institution ID field */}
        <Text style={styles.fieldLabel}>Institution ID</Text>
        <TextInput
          style={styles.input}
          placeholder="Enter your institution ID"
          placeholderTextColor={C.textMute}
          autoCapitalize="none"
          value={username}
          onChangeText={setUsername}
        />

        {/* Password field */}
        <Text style={styles.fieldLabel}>Password</Text>
        <TextInput
          style={styles.input}
          placeholder="Enter your password"
          placeholderTextColor={C.textMute}
          secureTextEntry
          value={password}
          onChangeText={setPassword}
        />

        {/* Forgot password — right-aligned text link */}
        <TouchableOpacity style={styles.forgotWrap}>
          <Text style={styles.forgotText}>Forgot password?</Text>
        </TouchableOpacity>

        {/* Primary navy Sign in button */}
        <TouchableOpacity
          style={styles.signInBtn}
          onPress={handleLogin}
          disabled={loading}>
          {loading ? (
            <ActivityIndicator color={C.textOnNavy} />
          ) : (
            <Text style={styles.signInBtnText}>Sign in</Text>
          )}
        </TouchableOpacity>
      </View>

      {/* ── Footer note ───────────────────────────────────────────────── */}
      <Text style={styles.footer}>
        New user? Contact your department admin
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: C.bg,
  },

  // ── Navy header block ────────────────────────────────────────────────────
  headerBlock: {
    backgroundColor: C.navy,
    paddingTop: 56,
    paddingBottom: 40,
    alignItems: 'center',
  },
  iconBadge: {
    width: 52,
    height: 52,
    borderRadius: RADIUS.chip,
    backgroundColor: C.surface,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 14,
  },
  iconBadgeText: {
    fontSize: 26,
  },
  appName: {
    color: C.textOnNavy,
    fontSize: 22,
    fontWeight: '700',
  },
  appSub: {
    color: C.navySub,
    fontSize: 13,
    marginTop: 4,
  },

  // ── White card ───────────────────────────────────────────────────────────
  card: {
    backgroundColor: C.surface,
    marginHorizontal: 20,
    marginTop: -20,
    borderRadius: RADIUS.card,
    borderWidth: 1,
    borderColor: C.border,
    padding: 24,
  },
  fieldLabel: {
    color: C.navy,
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 6,
  },
  input: {
    backgroundColor: C.surface,
    borderColor: C.border,
    borderWidth: 1,
    borderRadius: RADIUS.btn,
    paddingHorizontal: 14,
    paddingVertical: 11,
    color: C.textBody,
    fontSize: 14,
    marginBottom: 16,
  },
  forgotWrap: {
    alignSelf: 'flex-end',
    marginBottom: 20,
    marginTop: -8,
  },
  forgotText: {
    color: C.navy,
    fontSize: 12,
  },
  signInBtn: {
    backgroundColor: C.navy,
    paddingVertical: 14,
    borderRadius: RADIUS.btn,
    alignItems: 'center',
  },
  signInBtnText: {
    color: C.textOnNavy,
    fontWeight: '700',
    fontSize: 15,
  },

  // ── Footer ───────────────────────────────────────────────────────────────
  footer: {
    color: C.textMute,
    fontSize: 12,
    textAlign: 'center',
    marginTop: 20,
  },
});

export default LoginScreen;
