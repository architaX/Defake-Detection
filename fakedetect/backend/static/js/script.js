document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    let formData = new FormData();
    let file = document.getElementById('fileInput').files[0];
    let type = document.getElementById('typeInput').value;

    formData.append('file', file);
    formData.append('type', type);

    const res = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    if (!res.ok) {
        document.getElementById('result').innerText = 'Error uploading file.';
        return;
    }

    const data = await res.json();
    document.getElementById('result').innerText = `Result: ${data.result}`;  // Fixed: Added backticks
});