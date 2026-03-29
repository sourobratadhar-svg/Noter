/**
 * NOTES Screen - Ingest & Manage Notes
 * Paste text or upload .txt/.md files. View and delete ingested notes.
 * Terminal-style input area, flat card list.
 */
import { useState, useEffect, useCallback } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, FlatList,
  StyleSheet, SafeAreaView, Alert, ScrollView, Platform,
} from 'react-native';
import * as DocumentPicker from 'expo-document-picker';

const API_URL = process.env.EXPO_PUBLIC_BACKEND_URL;

type Note = {
  id: string;
  title: string;
  source_type: string;
  chunk_count: number;
  char_count: number;
  created_at: string;
};

export default function NotesScreen() {
  const [notes, setNotes] = useState<Note[]>([]);
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [mode, setMode] = useState<'list' | 'add'>('list');

  const fetchNotes = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await fetch(`${API_URL}/api/notes`);
      const data = await resp.json();
      setNotes(data);
    } catch {
      // silently fail
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchNotes(); }, []);

  const ingestText = async () => {
    if (!content.trim()) return;
    setIngesting(true);
    try {
      const resp = await fetch(`${API_URL}/api/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: title.trim(), content: content.trim(), source_type: 'paste' }),
      });
      const data = await resp.json();
      if (resp.ok) {
        Alert.alert('INGESTED', `${data.chunk_count} chunks indexed from "${data.title}"`);
        setTitle('');
        setContent('');
        setMode('list');
        fetchNotes();
      } else {
        Alert.alert('ERROR', data.detail || 'Ingestion failed');
      }
    } catch {
      Alert.alert('ERROR', 'Could not connect to backend');
    } finally {
      setIngesting(false);
    }
  };

  const pickFile = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ['text/plain', 'text/markdown', 'text/*'],
        copyToCacheDirectory: true,
      });

      if (result.canceled || !result.assets || result.assets.length === 0) return;

      const file = result.assets[0];
      const ext = file.name?.split('.').pop()?.toLowerCase();
      if (ext !== 'txt' && ext !== 'md') {
        Alert.alert('UNSUPPORTED', 'Only .txt and .md files are supported');
        return;
      }

      setIngesting(true);

      // Read file content and send as text
      const response = await fetch(file.uri);
      const text = await response.text();

      const resp = await fetch(`${API_URL}/api/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: file.name || 'Uploaded File',
          content: text,
          source_type: 'file',
        }),
      });
      const data = await resp.json();
      if (resp.ok) {
        Alert.alert('INGESTED', `${data.chunk_count} chunks indexed from "${data.title}"`);
        fetchNotes();
      } else {
        Alert.alert('ERROR', data.detail || 'File ingestion failed');
      }
    } catch (err) {
      Alert.alert('ERROR', 'File upload failed');
    } finally {
      setIngesting(false);
    }
  };

  const deleteNote = async (noteId: string, noteTitle: string) => {
    Alert.alert('DELETE NOTE', `Remove "${noteTitle}" and all its vectors?`, [
      { text: 'CANCEL', style: 'cancel' },
      {
        text: 'DELETE', style: 'destructive',
        onPress: async () => {
          try {
            await fetch(`${API_URL}/api/notes/${noteId}`, { method: 'DELETE' });
            fetchNotes();
          } catch {
            Alert.alert('ERROR', 'Could not delete note');
          }
        },
      },
    ]);
  };

  const renderNote = ({ item }: { item: Note }) => (
    <View style={styles.noteCard} testID={`note-card-${item.id}`}>
      <View style={styles.noteHeader}>
        <Text style={styles.noteTitle} numberOfLines={1}>{item.title}</Text>
        <TouchableOpacity
          testID={`note-delete-${item.id}`}
          onPress={() => deleteNote(item.id, item.title)}
          style={styles.deleteBtn}
          activeOpacity={0.7}
        >
          <Text style={styles.deleteBtnText}>DEL</Text>
        </TouchableOpacity>
      </View>
      <View style={styles.noteMeta}>
        <Text style={styles.metaText}>{item.source_type.toUpperCase()}</Text>
        <Text style={styles.metaDot}> // </Text>
        <Text style={styles.metaText}>{item.chunk_count} CHUNKS</Text>
        <Text style={styles.metaDot}> // </Text>
        <Text style={styles.metaText}>{item.char_count} CHARS</Text>
      </View>
      <Text style={styles.noteDate}>{new Date(item.created_at).toLocaleDateString()}</Text>
    </View>
  );

  if (mode === 'add') {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>ADD NOTE</Text>
          <TouchableOpacity testID="back-to-list-btn" onPress={() => setMode('list')} activeOpacity={0.7}>
            <Text style={styles.backBtn}>BACK</Text>
          </TouchableOpacity>
        </View>
        <ScrollView style={styles.addForm} keyboardShouldPersistTaps="handled">
          <Text style={styles.fieldLabel}>TITLE (OPTIONAL)</Text>
          <TextInput
            testID="note-title-input"
            style={styles.titleInput}
            value={title}
            onChangeText={setTitle}
            placeholder="Note title..."
            placeholderTextColor="#999"
          />

          <Text style={styles.fieldLabel}>CONTENT</Text>
          <TextInput
            testID="note-content-input"
            style={styles.contentInput}
            value={content}
            onChangeText={setContent}
            placeholder="Paste your notes here..."
            placeholderTextColor="#999"
            multiline
            textAlignVertical="top"
          />

          <TouchableOpacity
            testID="ingest-text-btn"
            style={[styles.primaryBtn, (!content.trim() || ingesting) && styles.btnDisabled]}
            onPress={ingestText}
            disabled={!content.trim() || ingesting}
            activeOpacity={0.7}
          >
            <Text style={styles.primaryBtnText}>
              {ingesting ? '[ INDEXING... ]' : '[ INGEST TEXT ]'}
            </Text>
          </TouchableOpacity>

          <View style={styles.divider}>
            <View style={styles.dividerLine} />
            <Text style={styles.dividerText}>OR</Text>
            <View style={styles.dividerLine} />
          </View>

          <TouchableOpacity
            testID="upload-file-btn"
            style={[styles.secondaryBtn, ingesting && styles.btnDisabled]}
            onPress={pickFile}
            disabled={ingesting}
            activeOpacity={0.7}
          >
            <Text style={styles.secondaryBtnText}>[ UPLOAD .TXT / .MD FILE ]</Text>
          </TouchableOpacity>
        </ScrollView>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>NOTES</Text>
          <Text style={styles.headerSub}>{notes.length} INDEXED</Text>
        </View>
        <TouchableOpacity
          testID="add-note-btn"
          style={styles.addBtn}
          onPress={() => setMode('add')}
          activeOpacity={0.7}
        >
          <Text style={styles.addBtnText}>+ ADD</Text>
        </TouchableOpacity>
      </View>

      <FlatList
        testID="notes-list"
        data={notes}
        renderItem={renderNote}
        keyExtractor={item => item.id}
        style={styles.notesList}
        contentContainerStyle={notes.length === 0 ? styles.emptyList : undefined}
        refreshing={loading}
        onRefresh={fetchNotes}
        ListEmptyComponent={
          <View style={styles.emptyContainer} testID="notes-empty-state">
            <Text style={styles.emptyTitle}>NO NOTES</Text>
            <Text style={styles.emptyHint}>Tap [+ ADD] to ingest your first note.</Text>
          </View>
        }
      />
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
  headerSub: {
    fontFamily: 'Courier', fontSize: 11, color: '#555555',
    letterSpacing: 2, marginTop: 2,
  },
  backBtn: {
    fontFamily: 'Courier', fontWeight: '800', fontSize: 14,
    color: '#002FA7', letterSpacing: 2,
  },
  addBtn: {
    backgroundColor: '#0A0A0A', paddingHorizontal: 16, paddingVertical: 10,
  },
  addBtnText: {
    fontFamily: 'Courier', fontWeight: '800', fontSize: 13,
    color: '#FFFFFF', letterSpacing: 2,
  },
  notesList: { flex: 1 },
  emptyList: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  noteCard: {
    borderBottomWidth: 1, borderBottomColor: '#0A0A0A',
    paddingHorizontal: 16, paddingVertical: 14,
  },
  noteHeader: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
  },
  noteTitle: {
    fontFamily: 'Courier', fontWeight: '700', fontSize: 15,
    color: '#0A0A0A', flex: 1, marginRight: 12,
  },
  deleteBtn: {
    borderWidth: 1, borderColor: '#FF2A00', paddingHorizontal: 10, paddingVertical: 4,
  },
  deleteBtnText: {
    fontFamily: 'Courier', fontWeight: '700', fontSize: 11,
    color: '#FF2A00', letterSpacing: 1,
  },
  noteMeta: {
    flexDirection: 'row', alignItems: 'center', marginTop: 6,
  },
  metaText: {
    fontFamily: 'Courier', fontSize: 11, color: '#555555', letterSpacing: 1,
  },
  metaDot: {
    fontFamily: 'Courier', fontSize: 11, color: '#CCCCCC',
  },
  noteDate: {
    fontFamily: 'Courier', fontSize: 10, color: '#999', marginTop: 4,
  },
  emptyContainer: { alignItems: 'center' },
  emptyTitle: {
    fontFamily: 'Courier', fontWeight: '900', fontSize: 24,
    color: '#0A0A0A',
  },
  emptyHint: {
    fontFamily: 'Courier', fontSize: 13, color: '#555555',
    marginTop: 8,
  },
  addForm: { flex: 1, paddingHorizontal: 16, paddingTop: 16 },
  fieldLabel: {
    fontFamily: 'Courier', fontWeight: '600', fontSize: 11,
    color: '#555555', letterSpacing: 2, marginBottom: 6,
  },
  titleInput: {
    fontFamily: 'Courier', fontSize: 14, color: '#0A0A0A',
    borderBottomWidth: 2, borderBottomColor: '#0A0A0A',
    paddingVertical: 10, marginBottom: 20,
  },
  contentInput: {
    fontFamily: 'Courier', fontSize: 13, color: '#0A0A0A',
    borderWidth: 2, borderColor: '#0A0A0A',
    padding: 12, minHeight: 200, marginBottom: 20,
    lineHeight: 20,
  },
  primaryBtn: {
    backgroundColor: '#0A0A0A', paddingVertical: 16, alignItems: 'center',
    borderWidth: 2, borderColor: '#0A0A0A',
  },
  primaryBtnText: {
    fontFamily: 'Courier', fontWeight: '800', fontSize: 14,
    color: '#FFFFFF', letterSpacing: 2,
  },
  secondaryBtn: {
    backgroundColor: '#FFFFFF', paddingVertical: 16, alignItems: 'center',
    borderWidth: 1, borderColor: '#0A0A0A',
  },
  secondaryBtnText: {
    fontFamily: 'Courier', fontWeight: '600', fontSize: 13,
    color: '#0A0A0A', letterSpacing: 1,
  },
  btnDisabled: { opacity: 0.4 },
  divider: {
    flexDirection: 'row', alignItems: 'center', marginVertical: 20,
  },
  dividerLine: { flex: 1, height: 1, backgroundColor: '#CCCCCC' },
  dividerText: {
    fontFamily: 'Courier', fontSize: 12, color: '#555555',
    marginHorizontal: 12, letterSpacing: 2,
  },
});
