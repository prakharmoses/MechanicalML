import { useState } from 'react';
import axios from 'axios';

// Importing styles
import '../styles/ImageUploadPopup.css';

const ImageUploadPopup = ({ open, onClose, onImagesSelected }) => {
    const [selectedFiles, setSelectedFiles] = useState([]);

    const handleFileChange = (event) => {
        setSelectedFiles([...event.target.files]);
    };

    const handleUpload = async () => {
        const formData = new FormData();
        selectedFiles.forEach((file) => {
            formData.append('files', file);
        })
        const imagePaths = Array.from(selectedFiles).map((file) => URL.createObjectURL(file));
        try {
            const response = await axios.post("http://127.0.0.1:8000/model/upload_multiple",
                formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            const downloadedPaths = response.data.output;
            onImagesSelected({imagePaths, downloadedPaths});
        } catch (error) {
            console.log('Upload failed', error)
        }
        onClose(); // Close the popup
    };

    return (
        <div className={`image-upload-popup ${open ? 'active' : ''}`}>
            <div className="popup-content">
                <h2>Upload Images</h2>
                <input type="file" accept="image/*" multiple onChange={handleFileChange} />
                <button onClick={handleUpload} disabled={!selectedFiles.length}>
                    Upload
                </button>
                <button onClick={onClose}>Close</button>
            </div>
        </div>
    );
};

export default ImageUploadPopup;
