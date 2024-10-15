// const { Firestore } = require('@google-cloud/firestore');
// const { storage, ref, uploadBytes, getDownloadURL } = require('../../firebase-config'); // Adjust the path as necessary

// // const db = new Firestore();

// const storeData = async (id, data) => {

//     const predictCollection = db.collection('predictions');
//     return predictCollection.doc(id).set(data);
// }

// const storeImage = async (buffer, filename) => {
//     try {
//         const storageRef = ref(storage, filename);
//         const metadata = {
//             contentType: 'image/jpg' 
//         };
//         await uploadBytes(storageRef, buffer, metadata);
//         const imageUrl = await getDownloadURL(storageRef);
//         return imageUrl;
//     } catch (error) {
//         console.error('Error uploading image to Firebase Storage:', error);
//         throw error;
//     }
// }

// const getDataByUserID = async(uid) => {
//     try {
//         const predictionsRef = db.collection('predictions');
//         const querySnapshot = await predictionsRef.where('userID', '==', uid).orderBy('createdAt', 'desc').get();;
      
//         if (querySnapshot.empty) {
//           console.log('No matching documents.');
//           return [];
//         }
      
//         let predictions = [];
//         querySnapshot.forEach(doc => {
//           predictions.push({ id: doc.id, ...doc.data() });
//         });
      
//         return predictions;
//       } catch (error) {
//         console.error('Error getting predictions:', error);
//         throw new Error('Failed to get predictions');
//       }
// }
// module.exports = { storeData, storeImage, getDataByUserID }; 
