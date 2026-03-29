/**
 * SYS Screen - System Health & Statistics
 * Shows ChromaDB status, Ollama status, embedding model info, stats.
 */
import { useState, useEffect, useCallback } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet, SafeAreaView, ScrollView,
} from 'react-native';

const API_URL = process.env.EXPO_PUBLIC_BACKEND_URL;

type HealthData = {
  status: string;
  chromadb: string;
  ollama: string;
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
  const [loading, setLoading] = useState(false);
  const [lastCheck, setLastCheck] = useState('');

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [hResp, sResp] = await Promise.all([
        fetch(`${API_URL}/api/health`),
        fetch(`${API_URL}/api/stats`),
      ]);
      const hData = await hResp.json();
      const sData = await sResp.json();
      setHealth(hData);
      setStats(sData);
      setLastCheck(new Date().toLocaleTimeString());
    } catch {
      setHealth(null);
      setStats(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, []);

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
        {/* Health Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>HEALTH STATUS</Text>
          {health ? (
            <>
              <StatusRow
                label="BACKEND"
                value={health.status.toUpperCase()}
                status={health.status === 'operational' ? 'ok' : 'err'}
              />
              <StatusRow
                label="CHROMADB"
                value={health.chromadb.toUpperCase()}
                status={health.chromadb === 'connected' ? 'ok' : 'err'}
              />
              <StatusRow
                label="OLLAMA"
                value={health.ollama.toUpperCase()}
                status={health.ollama === 'connected' ? 'ok' : 'warn'}
              />
              <StatusRow
                label="EMBEDDINGS"
                value={health.embedding_model}
                status="ok"
              />
            </>
          ) : (
            <View style={styles.errorBox} testID="sys-connection-error">
              <Text style={styles.errorText}>BACKEND UNREACHABLE</Text>
              <Text style={styles.errorSub}>Check if server is running on port 8001</Text>
            </View>
          )}
        </View>

        {/* Stats Section */}
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

        {/* Config Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>CONFIGURATION</Text>
          <StatusRow label="EMBEDDING MODEL" value={stats?.embedding_model || 'all-MiniLM-L6-v2'} />
          <StatusRow label="OLLAMA MODEL" value={stats?.ollama_model || 'mistral'} />
          <StatusRow label="VECTOR DB" value="ChromaDB (persistent)" />
          <StatusRow label="STORAGE" value="Local only" />
        </View>

        {/* Info Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>PRIVACY</Text>
          <View style={styles.infoBox}>
            <Text style={styles.infoText}>
              All data processing happens locally.{'\n'}
              No external API calls are made.{'\n'}
              Your notes never leave this device.{'\n'}
              Embeddings generated on-device.
            </Text>
          </View>
        </View>

        {/* Ollama Setup Help */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>OLLAMA SETUP</Text>
          <View style={styles.infoBox}>
            <Text style={styles.infoCode}>
              # Install Ollama{'\n'}
              curl -fsSL https://ollama.ai/install.sh | sh{'\n'}
              {'\n'}
              # Pull a model{'\n'}
              ollama pull mistral{'\n'}
              {'\n'}
              # Ollama runs on localhost:11434{'\n'}
              # The app auto-detects it
            </Text>
          </View>
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
  statusDot: {
    width: 8, height: 8, marginRight: 8,
  },
  statusValue: {
    fontFamily: 'Courier', fontWeight: '700', fontSize: 12,
    color: '#0A0A0A',
  },
  statsGrid: {
    flexDirection: 'row', justifyContent: 'space-between',
  },
  statBlock: {
    flex: 1, alignItems: 'center', paddingVertical: 16,
    borderWidth: 1, borderColor: '#0A0A0A', marginHorizontal: 4,
  },
  statValue: {
    fontFamily: 'Courier', fontWeight: '900', fontSize: 28,
    color: '#0A0A0A',
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
  infoBox: {
    backgroundColor: '#F5F5F5', borderWidth: 1, borderColor: '#E0E0E0',
    padding: 14,
  },
  infoText: {
    fontFamily: 'Courier', fontSize: 12, color: '#0A0A0A', lineHeight: 20,
  },
  infoCode: {
    fontFamily: 'Courier', fontSize: 11, color: '#0A0A0A', lineHeight: 18,
  },
  lastCheck: {
    fontFamily: 'Courier', fontSize: 10, color: '#999',
    textAlign: 'center', marginTop: 16, letterSpacing: 2,
  },
});
