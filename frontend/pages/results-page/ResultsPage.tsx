import { useRouter } from 'expo-router';
import { View, Text, Button } from 'react-native';

export default function ResultsPage() {
  const router = useRouter();

  function handlePress(): void {
    router.push('/');
  }

  return (
    <View
      style={{
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <Text>Results Page</Text>
      <Text>Your cat is fat</Text>
      <Button title="Restart" onPress={handlePress} />
    </View>
  );
}
