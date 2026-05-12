const form = document.querySelector("#upload-form");
const imageInput = document.querySelector("#image-input");
const fileNameEl = document.querySelector("#file-name");
const submitBtn = document.querySelector("#submit-btn");
const statusEl = document.querySelector("#status");
const resultEl = document.querySelector("#result");

const freeSeatsEl = document.querySelector("#free-seats");
const occupiedSeatsEl = document.querySelector("#occupied-seats");
const totalSeatsEl = document.querySelector("#total-seats");
const desksDetectedEl = document.querySelector("#desks-detected");
const deskTableBody = document.querySelector("#desk-table-body");
const annotatedImage = document.querySelector("#annotated-image");
const noteEl = document.querySelector("#note");

function seatText(seats, side) {
  const seat = seats.find((item) => item.side === side);
  if (!seat) {
    return "-";
  }
  return seat.occupied ? "занято" : "свободно";
}

function renderResult(data) {
  freeSeatsEl.textContent = data.free_seats;
  occupiedSeatsEl.textContent = data.occupied_seats;
  totalSeatsEl.textContent = data.total_seats;
  desksDetectedEl.textContent = data.desks_detected;
  noteEl.textContent = data.note || "";
  annotatedImage.src = data.annotated_image;

  deskTableBody.innerHTML = "";
  for (const desk of data.desks) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${desk.index}</td>
      <td>${desk.source}</td>
      <td>${seatText(desk.seats, "left")}</td>
      <td>${seatText(desk.seats, "right")}</td>
      <td>${desk.free_count}</td>
    `;
    deskTableBody.appendChild(row);
  }

  resultEl.hidden = false;
}

imageInput.addEventListener("change", () => {
  const [file] = imageInput.files;
  if (file) {
    fileNameEl.textContent = file.name;
    submitBtn.disabled = false;
    statusEl.textContent = "";
  } else {
    fileNameEl.textContent = "Файл не выбран";
    submitBtn.disabled = true;
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const [file] = imageInput.files;
  if (!file) {
    statusEl.textContent = "Сначала выберите файл.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  statusEl.className = "";
  statusEl.textContent = "Идет обработка изображения...";
  resultEl.hidden = true;
  submitBtn.disabled = true;

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Ошибка обработки");
    }

    statusEl.textContent = "Готово.";
    statusEl.className = "status-success";
    renderResult(data);
  } catch (error) {
    statusEl.textContent = `Ошибка: ${error.message}`;
    statusEl.className = "status-error";
  } finally {
    submitBtn.disabled = false;
  }
});
