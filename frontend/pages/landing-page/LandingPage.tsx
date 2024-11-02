import { router, useRouter } from 'expo-router';
import { View, Text, Button, Alert } from 'react-native';

export default function LandingPage() {
  const router = useRouter();

  function handlePress(): void {
    router.push('/upload');
  }

  return (
    <View
      style={{
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <Text>Landing Page</Text>
      <Button onPress={handlePress} title="Let's go!" />
    </View>
  );
}
