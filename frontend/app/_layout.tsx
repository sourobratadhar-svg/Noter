/**
 * Tab Layout — Bottom navigation for the RAG Notes app.
 * 4 tabs: CHAT, NOTES, GRAPH, SYS
 * Brutalist Swiss design: uppercase mono text, no icons, blue active indicator.
 */
import { Tabs } from 'expo-router';
import { View, Text, StyleSheet } from 'react-native';

function TabLabel({ label, focused }: { label: string; focused: boolean }) {
  return (
    <View style={styles.tabItem}>
      <Text style={[styles.tabText, focused && styles.tabTextActive]}>
        [ {label} ]
      </Text>
      {focused && <View style={styles.tabIndicator} />}
    </View>
  );
}

export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarStyle: styles.tabBar,
        tabBarShowLabel: false,
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          tabBarIcon: ({ focused }) => <TabLabel label="CHAT" focused={focused} />,
        }}
      />
      <Tabs.Screen
        name="notes"
        options={{
          tabBarIcon: ({ focused }) => <TabLabel label="NOTES" focused={focused} />,
        }}
      />
      <Tabs.Screen
        name="graph"
        options={{
          tabBarIcon: ({ focused }) => <TabLabel label="GRAPH" focused={focused} />,
        }}
      />
      <Tabs.Screen
        name="system"
        options={{
          tabBarIcon: ({ focused }) => <TabLabel label="SYS" focused={focused} />,
        }}
      />
    </Tabs>
  );
}

const styles = StyleSheet.create({
  tabBar: {
    backgroundColor: '#FFFFFF',
    borderTopWidth: 2,
    borderTopColor: '#0A0A0A',
    height: 64,
    paddingTop: 8,
    paddingBottom: 8,
    elevation: 0,
    shadowOpacity: 0,
  },
  tabItem: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  tabText: {
    fontFamily: 'Courier',
    fontWeight: '600',
    fontSize: 11,
    color: '#555555',
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  tabTextActive: {
    color: '#002FA7',
    fontWeight: '800',
  },
  tabIndicator: {
    width: 32,
    height: 3,
    backgroundColor: '#002FA7',
    marginTop: 3,
  },
});
