import { useState } from 'react';
import axios from 'axios';

// Importing components
import ImageUploadPopup from './ImageUploadPopup';

// Importing styles
import '../styles/SolarPanelFaultDetect.css';

const SolarPanelFaultDetect = () => {
    const [isPopupOpen, setIsPopupOpen] = useState(false);
    const [selectedImages, setSelectedImages] = useState([]);
    const [downloadedImages, setDownLoadedImages] = useState([])
    const [error, setError] = useState(null);
    const [result, setResult] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleOpenPopup = () => {
        setResult([]);
        setIsPopupOpen(true);
    };

    const handleImagesSelected = async ({ imagePaths, downloadedPaths }) => {
        setSelectedImages(() => ([...selectedImages, ...imagePaths]));
        setDownLoadedImages([...downloadedImages, ...downloadedPaths]);
    };

    const handlePopupClose = () => {
        setIsPopupOpen(false);
    };

    const handleClassifyImg = async () => {
        setLoading(true);
        try {
            const response = await axios.post('http://127.0.0.1:8000/model/solar_panel', {
                img_paths: downloadedImages
            });
            const type = response.data.output;
            setResult(type);
            setDownLoadedImages([])
        } catch (error) {
            setError(error.message)
        } finally {
            setLoading(false);
        }
    }

    const handleModelRetrain = async () => {
        setResult([])
        try {
            setLoading(true);
            const response = await axios.get('http://127.0.0.1:8000/model/solar_panel/train');
            const output = response.data.output;
            console.log(output);
            // setResult(output)
        } catch (error) {
            setError(error.message)
        } finally {
            setTimeout(() => {
                setLoading(false);
            }, 1800000);
        }
    }

    const handleClear = async () => {
        try {
            await axios.post('http://127.0.0.1:8000/model/clear_all', downloadedImages);
        } catch (error) {
            setError(error.message);
        }
        setSelectedImages([]);
        setResult([]);
        setDownLoadedImages([]);
    }

    return (
        <main className='solarpanel-container'>
            <h1>Welcome to Fault Detection in Solar Panels</h1>
            <button onClick={handleOpenPopup}>{loading ? "Loading..." : "Open Image Upload"}</button>
            {selectedImages.length === 0 && (
                <button onClick={handleModelRetrain}>{loading ? "Loading..." : "Retrain the Model"}</button>
            )}

            {selectedImages.length > 0 && result.length === 0 && (
                <div>
                    Selected Images:
                    <div className='image-grid'>
                        {selectedImages.map((path) => (
                            <img key={path} src={path} alt='Seleted' />
                        ))}
                    </div>
                </div>
            )}

            {isPopupOpen && <ImageUploadPopup
                open={isPopupOpen}
                onClose={handlePopupClose}
                onImagesSelected={handleImagesSelected}
            />}
            {downloadedImages.length > 0 && (
                <button onClick={handleClassifyImg} disabled={loading}>{loading ? 'Loading...' : 'Classify'}</button>
            )}

            {result.length > 0 && !error && (
                <div className='image-grid'>
                    {result.map((res, idx) => (
                        <div key={idx}>
                            <img src={selectedImages[idx]} alt="Selected" />
                            <p>{res}</p>
                        </div>
                    ))}
                </div>
            )}

            {(result.length > 0 || selectedImages.length > 0) && !error && (
                <button onClick={handleClear}>Clear</button>
            )}

            {error && <p className='error'>{error}</p>}
        </main>
    );
}

export default SolarPanelFaultDetect;