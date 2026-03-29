/**
 * SYS Screen — System Health, Ollama Diagnostics & Network Setup
 * ===============================================================
 * Shows detailed health of all local services.
 * Provides Ollama setup instructions and model management.
 * Includes network setup guide for mobile-to-laptop connection.
 */
import { useState, useEffect, useCallback } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet, ScrollView, TextInput,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

const API_URL = process.env.EXPO_PUBLIC_BACKEND_URL;

type HealthData = {
  status: string;
  chromadb: string;
  ollama: string;
  ollama_error: string | null;
  ollama_models: string[];
  ollama_active_model: string;
  ollama_model_loaded: boolean;
  embedding_model: string;
  total_chunks: number;
};

type StatsData = {
  total_notes: number;
  total_chunks: number;
  embedding_model: string;
  embedding_dim: number;
  ollama_model: string;
};

type OllamaStatus = {
  available: boolean;
  models: string[];
  active_model: string;
  model_loaded: boolean;
  error: string | null;
  base_url: string;
  troubleshooting: { not_running: string; no_model: string; network: string };
};

function StatusRow({ label, value, status }: { label: string; value: string; status?: 'ok' | 'warn' | 'err' }) {
  const statusColor = status === 'ok' ? '#00D154' : status === 'err' ? '#FF2A00' : status === 'warn' ? '#FFDE00' : '#555555';
  return (
    <View style={styles.statusRow} testID={`sys-status-${label.toLowerCase().replace(/\s/g, '-')}`}>
      <Text style={styles.statusLabel}>{label}</Text>
      <View style={styles.statusValueRow}>
        {status && <View style={[styles.statusDot, { backgroundColor: statusColor }]} />}
        <Text style={[styles.statusValue, status === 'err' && { color: '#FF2A00' }]}>{value}</Text>
      </View>
    </View>
  );
}

function StatBlock({ label, value }: { label: string; value: string | number }) {
  return (
    <View style={styles.statBlock} testID={`sys-stat-${label.toLowerCase().replace(/\s/g, '-')}`}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

export default function SystemScreen() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [stats, setStats] = useState<StatsData | null>(null);
  const [ollamaDetail, setOllamaDetail] = useState<OllamaStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastCheck, setLastCheck] = useState('');
  const [modelInput, setModelInput] = useState('');
  const [modelMsg, setModelMsg] = useState('');

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [hResp, sResp, oResp] = await Promise.all([
        fetch(`${API_URL}/api/health`),
        fetch(`${API_URL}/api/stats`),
        fetch(`${API_URL}/api/ollama/status`),
      ]);
      setHealth(await hResp.json());
      setStats(await sResp.json());
      setOllamaDetail(await oResp.json());
      setLastCheck(new Date().toLocaleTimeString());
    } catch {
      setHealth(null);
      setStats(null);
      setOllamaDetail(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, []);

  const switchModel = async () => {
    const m = modelInput.trim();
    if (!m) return;
    try {
      const resp = await fetch(`${API_URL}/api/ollama/model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: m }),
      });
      const data = await resp.json();
      setModelMsg(data.message || 'Model updated');
      setModelInput('');
      refresh();
    } catch {
      setModelMsg('Failed to update model');
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>SYSTEM</Text>
        <TouchableOpacity
          testID="sys-refresh-btn"
          style={styles.refreshBtn}
          onPress={refresh}
          disabled={loading}
          activeOpacity={0.7}
        >
          <Text style={styles.refreshBtnText}>
            {loading ? '[ ... ]' : '[ REFRESH ]'}
          </Text>
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.content}>
        {/* ── Health Status ── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>HEALTH STATUS</Text>
          {health ? (
            <>
              <StatusRow label="BACKEND" value={health.status.toUpperCase()} status={health.status === 'operational' ? 'ok' : 'err'} />
              <StatusRow label="CHROMADB" value={health.chromadb.toUpperCase()} status={health.chromadb === 'connected' ? 'ok' : 'err'} />
              <StatusRow label="OLLAMA" value={health.ollama.toUpperCase()} status={health.ollama === 'connected' ? 'ok' : 'warn'} />
              <StatusRow label="EMBEDDINGS" value={health.embedding_model} status="ok" />
            </>
          ) : (
            <View style={styles.errorBox} testID="sys-connection-error">
              <Text style={styles.errorText}>BACKEND UNREACHABLE</Text>
              <Text style={styles.errorSub}>Ensure backend is running on port 8001</Text>
              <Text style={styles.errorSub}>Phone and laptop must be on same WiFi</Text>
            </View>
          )}
        </View>

        {/* ── Ollama Details ── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>OLLAMA LLM</Text>
          {ollamaDetail ? (
            <>
              <StatusRow
                label="CONNECTION"
                value={ollamaDetail.available ? 'ONLINE' : 'OFFLINE'}
                status={ollamaDetail.available ? 'ok' : 'err'}
              />
              <StatusRow
                label="ACTIVE MODEL"
                value={ollamaDetail.active_model}
                status={ollamaDetail.model_loaded ? 'ok' : 'warn'}
              />
              {ollamaDetail.models.length > 0 && (
                <StatusRow
                  label="AVAILABLE"
                  value={ollamaDetail.models.join(', ')}
                />
              )}
              {ollamaDetail.error && (
                <View style={styles.warningBox} testID="ollama-error-detail">
                  <Text style={styles.warningText}>{ollamaDetail.error}</Text>
                </View>
              )}

              {/* Model switcher */}
              <View style={styles.modelSwitcher}>
                <TextInput
                  testID="model-input"
                  style={styles.modelInput}
                  value={modelInput}
                  onChangeText={setModelInput}
                  placeholder="e.g. mistral, llama3, phi3"
                  placeholderTextColor="#999"
                />
                <TouchableOpacity
                  testID="switch-model-btn"
                  style={[styles.modelBtn, !modelInput.trim() && styles.btnDisabled]}
                  onPress={switchModel}
                  disabled={!modelInput.trim()}
                  activeOpacity={0.7}
                >
                  <Text style={styles.modelBtnText}>SET</Text>
                </TouchableOpacity>
              </View>
              {modelMsg ? <Text style={styles.modelMsg}>{modelMsg}</Text> : null}
            </>
          ) : health ? (
            <Text style={styles.infoTextSmall}>Could not fetch Ollama details</Text>
          ) : null}
        </View>

        {/* ── Statistics ── */}
        {stats && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>STATISTICS</Text>
            <View style={styles.statsGrid}>
              <StatBlock label="NOTES" value={stats.total_notes} />
              <StatBlock label="CHUNKS" value={stats.total_chunks} />
              <StatBlock label="EMB DIM" value={stats.embedding_dim} />
            </View>
          </View>
        )}

        {/* ── Network Setup ── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>NETWORK SETUP</Text>
          <View style={styles.alertBox} testID="network-instructions">
            <Text style={styles.alertTitle}>MOBILE CONNECTION</Text>
            <Text style={styles.infoText}>
              Phone and laptop MUST be on the same{'\n'}
              WiFi network for the app to work.
            </Text>
          </View>
          <View style={styles.infoBox}>
            <Text style={styles.infoCode}>
              # Find your laptop's local IP:{'\n'}
              # macOS: ifconfig | grep inet{'\n'}
              # Linux: ip addr show | grep inet{'\n'}
              # Windows: ipconfig{'\n'}
              {'\n'}
              # Set backend URL in the app to:{'\n'}
              # http://YOUR_IP:8001{'\n'}
              {'\n'}
              # Example: http://192.168.1.42:8001
            </Text>
          </View>
        </View>

        {/* ── Ollama Setup ── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>OLLAMA SETUP GUIDE</Text>
          <View style={styles.infoBox}>
            <Text style={styles.infoCode}>
              # 1. Install Ollama{'\n'}
              curl -fsSL https://ollama.ai/install.sh | sh{'\n'}
              {'\n'}
              # 2. Pull a model (~4GB for mistral){'\n'}
              ollama pull mistral{'\n'}
              {'\n'}
              # 3. Start Ollama (auto-starts on install){'\n'}
              ollama serve{'\n'}
              {'\n'}
              # 4. Verify it works{'\n'}
              curl http://localhost:11434/api/tags{'\n'}
              {'\n'}
              # Other models to try:{'\n'}
              # ollama pull llama3{'\n'}
              # ollama pull phi3{'\n'}
              # ollama pull gemma2
            </Text>
          </View>
        </View>

        {/* ── Privacy ── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>PRIVACY</Text>
          <View style={styles.infoBox}>
            <Text style={styles.infoText}>
              All data processing happens locally.{'\n'}
              No external API calls are made.{'\n'}
              Your notes never leave this device.{'\n'}
              Embeddings + LLM run on your machine.
            </Text>
          </View>
        </View>

        {/* ── Configuration ── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>CONFIGURATION</Text>
          <StatusRow label="EMBEDDING MODEL" value={stats?.embedding_model || 'all-MiniLM-L6-v2'} />
          <StatusRow label="OLLAMA MODEL" value={stats?.ollama_model || 'mistral'} />
          <StatusRow label="VECTOR DB" value="ChromaDB (persistent)" />
          <StatusRow label="STORAGE" value="Local only" />
          <StatusRow label="BACKEND URL" value={API_URL || 'not set'} />
        </View>

        {lastCheck ? (
          <Text style={styles.lastCheck}>LAST CHECK: {lastCheck}</Text>
        ) : null}

        <View style={{ height: 40 }} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#FFFFFF' },
  header: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingHorizontal: 16, paddingTop: 16, paddingBottom: 12,
    borderBottomWidth: 2, borderBottomColor: '#0A0A0A',
  },
  headerTitle: {
    fontFamily: 'Courier', fontWeight: '900', fontSize: 28,
    color: '#0A0A0A', letterSpacing: -1,
  },
  refreshBtn: {
    borderWidth: 1, borderColor: '#0A0A0A', paddingHorizontal: 12, paddingVertical: 8,
  },
  refreshBtnText: {
    fontFamily: 'Courier', fontWeight: '600', fontSize: 12,
    color: '#0A0A0A', letterSpacing: 1,
  },
  content: { flex: 1 },
  section: {
    paddingHorizontal: 16, paddingTop: 20, paddingBottom: 12,
    borderBottomWidth: 1, borderBottomColor: '#E0E0E0',
  },
  sectionTitle: {
    fontFamily: 'Courier', fontWeight: '800', fontSize: 12,
    color: '#002FA7', letterSpacing: 3, marginBottom: 12,
  },
  statusRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: '#F5F5F5',
  },
  statusLabel: {
    fontFamily: 'Courier', fontWeight: '600', fontSize: 12,
    color: '#555555', letterSpacing: 1,
  },
  statusValueRow: { flexDirection: 'row', alignItems: 'center' },
  statusDot: { width: 8, height: 8, marginRight: 8 },
  statusValue: {
    fontFamily: 'Courier', fontWeight: '700', fontSize: 12,
    color: '#0A0A0A',
  },
  statsGrid: { flexDirection: 'row', justifyContent: 'space-between' },
  statBlock: {
    flex: 1, alignItems: 'center', paddingVertical: 16,
    borderWidth: 1, borderColor: '#0A0A0A', marginHorizontal: 4,
  },
  statValue: {
    fontFamily: 'Courier', fontWeight: '900', fontSize: 28, color: '#0A0A0A',
  },
  statLabel: {
    fontFamily: 'Courier', fontWeight: '600', fontSize: 10,
    color: '#555555', letterSpacing: 2, marginTop: 4,
  },
  errorBox: {
    borderWidth: 2, borderColor: '#FF2A00', padding: 16, alignItems: 'center',
  },
  errorText: {
    fontFamily: 'Courier', fontWeight: '800', fontSize: 14,
    color: '#FF2A00', letterSpacing: 2,
  },
  errorSub: {
    fontFamily: 'Courier', fontSize: 11, color: '#555555', marginTop: 4,
  },
  warningBox: {
    backgroundColor: '#FFF8E1', borderWidth: 1, borderColor: '#FFDE00',
    padding: 10, marginTop: 8,
  },
  warningText: {
    fontFamily: 'Courier', fontSize: 11, color: '#0A0A0A', lineHeight: 16,
  },
  alertBox: {
    backgroundColor: '#F0F4FF', borderWidth: 2, borderColor: '#002FA7',
    padding: 14, marginBottom: 12,
  },
  alertTitle: {
    fontFamily: 'Courier', fontWeight: '800', fontSize: 12,
    color: '#002FA7', letterSpacing: 2, marginBottom: 6,
  },
  infoBox: {
    backgroundColor: '#F5F5F5', borderWidth: 1, borderColor: '#E0E0E0', padding: 14,
  },
  infoText: {
    fontFamily: 'Courier', fontSize: 12, color: '#0A0A0A', lineHeight: 20,
  },
  infoTextSmall: {
    fontFamily: 'Courier', fontSize: 11, color: '#555555',
  },
  infoCode: {
    fontFamily: 'Courier', fontSize: 11, color: '#0A0A0A', lineHeight: 18,
  },
  modelSwitcher: {
    flexDirection: 'row', marginTop: 12,
  },
  modelInput: {
    flex: 1, fontFamily: 'Courier', fontSize: 13, color: '#0A0A0A',
    borderWidth: 2, borderColor: '#0A0A0A', paddingHorizontal: 10, paddingVertical: 8,
  },
  modelBtn: {
    backgroundColor: '#0A0A0A', paddingHorizontal: 16, justifyContent: 'center',
  },
  modelBtnText: {
    fontFamily: 'Courier', fontWeight: '800', fontSize: 13,
    color: '#FFFFFF', letterSpacing: 1,
  },
  btnDisabled: { opacity: 0.4 },
  modelMsg: {
    fontFamily: 'Courier', fontSize: 10, color: '#002FA7',
    marginTop: 6, letterSpacing: 1,
  },
  lastCheck: {
    fontFamily: 'Courier', fontSize: 10, color: '#999',
    textAlign: 'center', marginTop: 16, letterSpacing: 2,
  },
});
