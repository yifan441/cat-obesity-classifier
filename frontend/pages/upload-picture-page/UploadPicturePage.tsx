import { useRouter } from 'expo-router';
import { View, Text, Button } from 'react-native';

export default function UploadPicturePage() {
  const router = useRouter();

  function handlePress() {
    router.push('/results');
  }
  return (
    <View
      style={{
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <Text>Upload Picture Page</Text>
      <Button title="Upload" onPress={handlePress} />
    </View>
  );
}
