/**
 * CHAT Screen - RAG Query Interface
 * Ask questions about your notes, see AI answers with source context.
 * Brutalist Swiss design: full-width messages, no bubbles, monospace text.
 */
import { useState, useRef, useCallback } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, FlatList,
  StyleSheet, KeyboardAvoidingView, Platform, ActivityIndicator,
  Keyboard,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

const API_URL = process.env.EXPO_PUBLIC_BACKEND_URL;

type Source = {
  chunk_index: number;
  text: string;
  note_title: string;
  relevance: number;
};

type Message = {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  sources?: Source[];
  ollama_available?: boolean;
  ollama_error?: string;
  mode?: string;
  model?: string;
};

export default function ChatScreen() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const flatListRef = useRef<FlatList>(null);

  const askQuestion = useCallback(async () => {
    const q = input.trim();
    if (!q || loading) return;
    Keyboard.dismiss();

    const userMsg: Message = { id: Date.now().toString(), role: 'user', text: q };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const resp = await fetch(`${API_URL}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, top_k: 5 }),
      });
      const data = await resp.json();

      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        text: data.answer || 'No response.',
        sources: data.sources || [],
        ollama_available: data.ollama_available,
        ollama_error: data.ollama_error,
        mode: data.mode,
        model: data.model,
      };
      setMessages(prev => [...prev, aiMsg]);
    } catch (err) {
      const errMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        text: 'Cannot reach backend server.\n\nTroubleshooting:\n1. Is the backend running? (python server.py)\n2. Are phone and laptop on the same WiFi?\n3. Check the SYS tab for connection details.',
      };
      setMessages(prev => [...prev, errMsg]);
    } finally {
      setLoading(false);
    }
  }, [input, loading]);

  const renderSource = (source: Source, index: number) => (
    <View key={index} style={styles.sourceBlock} testID={`source-snippet-${index}`}>
      <Text style={styles.sourceLabel}>SRC // {source.note_title}</Text>
      <Text style={styles.sourceText}>{source.text}</Text>
      <Text style={styles.sourceRelevance}>RELEVANCE: {(source.relevance * 100).toFixed(1)}%</Text>
    </View>
  );

  const renderMessage = ({ item }: { item: Message }) => {
    const isUser = item.role === 'user';
    return (
      <View
        style={[styles.messageRow, isUser ? styles.userRow : styles.aiRow]}
        testID={`chat-message-${item.role}-${item.id}`}
      >
        <Text style={styles.roleLabel}>{isUser ? '> YOU' : '< RAG'}</Text>
        <Text style={styles.messageText}>{item.text}</Text>
        {item.sources && item.sources.length > 0 && (
          <View style={styles.sourcesContainer}>
            <Text style={styles.sourcesHeader}>SOURCES [{item.sources.length}]</Text>
            {item.sources.map((s, i) => renderSource(s, i))}
          </View>
        )}
        {item.role === 'assistant' && item.ollama_available === false && item.mode === 'extractive' && (
          <Text style={styles.fallbackNotice}>// EXTRACTIVE MODE — OLLAMA OFFLINE</Text>
        )}
        {item.role === 'assistant' && item.ollama_available === true && item.mode === 'ollama' && (
          <Text style={styles.ollamaNotice}>// OLLAMA [{item.model}]</Text>
        )}
      </View>
    );
  };

  const renderEmpty = () => (
    <View style={styles.emptyContainer} testID="chat-empty-state">
      <Text style={styles.emptyTitle}>LOCAL RAG</Text>
      <Text style={styles.emptySubtitle}>PRIVACY-FIRST NOTES RETRIEVAL</Text>
      <View style={styles.emptyDivider} />
      <Text style={styles.emptyHint}>Add notes in the [NOTES] tab, then ask questions here.</Text>
      <Text style={styles.emptyHint}>All processing happens locally on your device.</Text>
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header} testID="chat-header">
        <Text style={styles.headerTitle}>RAG QUERY</Text>
        <Text style={styles.headerSub}>LOCAL • PRIVATE • OFFLINE</Text>
      </View>

      <KeyboardAvoidingView
        style={styles.flex1}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={0}
      >
        <FlatList
          ref={flatListRef}
          data={messages}
          renderItem={renderMessage}
          keyExtractor={item => item.id}
          style={styles.messageList}
          contentContainerStyle={messages.length === 0 ? styles.emptyList : styles.listContent}
          ListEmptyComponent={renderEmpty}
          onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
        />

        {loading && (
          <View style={styles.loadingBar} testID="chat-loading">
            <Text style={styles.loadingText}>[ PROCESSING... ]</Text>
          </View>
        )}

        <View style={styles.inputBar} testID="chat-input-bar">
          <TextInput
            testID="chat-input"
            style={styles.textInput}
            value={input}
            onChangeText={setInput}
            placeholder="ASK YOUR NOTES..."
            placeholderTextColor="#999"
            returnKeyType="send"
            onSubmitEditing={askQuestion}
            editable={!loading}
          />
          <TouchableOpacity
            testID="chat-send-btn"
            style={[styles.sendBtn, (!input.trim() || loading) && styles.sendBtnDisabled]}
            onPress={askQuestion}
            disabled={!input.trim() || loading}
            activeOpacity={0.7}
          >
            <Text style={styles.sendBtnText}>ASK</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#FFFFFF' },
  flex1: { flex: 1 },
  header: {
    paddingHorizontal: 16, paddingTop: 16, paddingBottom: 12,
    borderBottomWidth: 2, borderBottomColor: '#0A0A0A',
  },
  headerTitle: {
    fontFamily: 'Courier', fontWeight: '900', fontSize: 28,
    color: '#0A0A0A', letterSpacing: -1,
  },
  headerSub: {
    fontFamily: 'Courier', fontSize: 11, color: '#555555',
    letterSpacing: 2, marginTop: 2,
  },
  messageList: { flex: 1 },
  listContent: { paddingBottom: 8 },
  emptyList: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  messageRow: {
    paddingHorizontal: 16, paddingVertical: 14,
    borderTopWidth: 1, borderTopColor: '#0A0A0A',
  },
  userRow: { backgroundColor: '#FFFFFF' },
  aiRow: { backgroundColor: '#F5F5F5' },
  roleLabel: {
    fontFamily: 'Courier', fontWeight: '800', fontSize: 11,
    color: '#002FA7', letterSpacing: 2, marginBottom: 6,
  },
  messageText: {
    fontFamily: 'Courier', fontSize: 14, color: '#0A0A0A',
    lineHeight: 22,
  },
  sourcesContainer: {
    marginTop: 12, borderTopWidth: 1, borderTopColor: '#CCCCCC',
    paddingTop: 10,
  },
  sourcesHeader: {
    fontFamily: 'Courier', fontWeight: '800', fontSize: 11,
    color: '#0A0A0A', letterSpacing: 2, marginBottom: 8,
  },
  sourceBlock: {
    backgroundColor: '#E0E0E0', borderWidth: 1, borderColor: '#0A0A0A',
    padding: 10, marginBottom: 6,
  },
  sourceLabel: {
    fontFamily: 'Courier', fontWeight: '600', fontSize: 10,
    color: '#002FA7', letterSpacing: 1, marginBottom: 4,
  },
  sourceText: {
    fontFamily: 'Courier', fontSize: 12, color: '#0A0A0A', lineHeight: 18,
  },
  sourceRelevance: {
    fontFamily: 'Courier', fontSize: 10, color: '#555555',
    marginTop: 4, letterSpacing: 1,
  },
  fallbackNotice: {
    fontFamily: 'Courier', fontSize: 10, color: '#FF2A00',
    marginTop: 8, letterSpacing: 1,
  },
  ollamaNotice: {
    fontFamily: 'Courier', fontSize: 10, color: '#00D154',
    marginTop: 8, letterSpacing: 1,
  },
  emptyContainer: { alignItems: 'center', paddingHorizontal: 32 },
  emptyTitle: {
    fontFamily: 'Courier', fontWeight: '900', fontSize: 36,
    color: '#0A0A0A', letterSpacing: -2,
  },
  emptySubtitle: {
    fontFamily: 'Courier', fontSize: 11, color: '#002FA7',
    letterSpacing: 3, marginTop: 4,
  },
  emptyDivider: {
    width: 60, height: 3, backgroundColor: '#0A0A0A', marginVertical: 20,
  },
  emptyHint: {
    fontFamily: 'Courier', fontSize: 13, color: '#555555',
    textAlign: 'center', lineHeight: 20, marginBottom: 4,
  },
  loadingBar: {
    backgroundColor: '#0A0A0A', paddingVertical: 8, alignItems: 'center',
  },
  loadingText: {
    fontFamily: 'Courier', fontWeight: '600', fontSize: 12,
    color: '#FFFFFF', letterSpacing: 2,
  },
  inputBar: {
    flexDirection: 'row', borderTopWidth: 2, borderTopColor: '#0A0A0A',
    backgroundColor: '#FFFFFF',
  },
  textInput: {
    flex: 1, fontFamily: 'Courier', fontSize: 14, color: '#0A0A0A',
    paddingHorizontal: 16, paddingVertical: 14,
  },
  sendBtn: {
    backgroundColor: '#0A0A0A', paddingHorizontal: 24,
    justifyContent: 'center', alignItems: 'center',
  },
  sendBtnDisabled: { backgroundColor: '#CCCCCC' },
  sendBtnText: {
    fontFamily: 'Courier', fontWeight: '800', fontSize: 14,
    color: '#FFFFFF', letterSpacing: 2,
  },
});
