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
const seatingChartEl = document.querySelector("#seating-chart");

function seatText(seats, side) {
  const seat = seats.find((item) => item.side === side);
  if (!seat) {
    return "-";
  }
  return seat.occupied ? "занято" : "свободно";
}

function renderSeatingChart(desks) {
  if (!desks || desks.length === 0) {
    seatingChartEl.innerHTML = '<p class="muted">Нет данных для отображения плана.</p>';
    return;
  }

  // Подготовка: центры и размеры
  const items = desks.map((d, idx) => {
    const cy = (d.box.y1 + d.box.y2) / 2;
    const cx = (d.box.x1 + d.box.x2) / 2;
    const h = d.box.y2 - d.box.y1;
    return { desk: d, idx, cy, cx, h };
  });

  // Сортируем сверху вниз (по Y)
  items.sort((a, b) => a.cy - b.cy);

  // Группировка по рядам на основе средней высоты парты
  const rows = [];
  let currentRow = [];
  let rowCenterY = 0;
  const avgH = items.reduce((s, it) => s + it.h, 0) / items.length;
  const threshold = avgH * 0.6;

  for (const it of items) {
    if (currentRow.length === 0 || Math.abs(it.cy - rowCenterY) <= threshold) {
      currentRow.push(it);
      rowCenterY = currentRow.reduce((s, x) => s + x.cy, 0) / currentRow.length;
    } else {
      rows.push(currentRow);
      currentRow = [it];
      rowCenterY = it.cy;
    }
  }
  if (currentRow.length) rows.push(currentRow);

  // Внутри каждого ряда сортируем слева направо (по X)
  rows.forEach((row) => row.sort((a, b) => a.cx - b.cx));

  // Генерация HTML
  let html = '';
  rows.forEach((row, ri) => {
    html += `<div class="seating-row">`;
    html += `<div class="row-label">Ряд ${ri + 1}</div>`;
    html += `<div class="row-seats">`;
    row.forEach((it, deskIdx) => {
      // Места в парте: сначала левое, потом правое
      const seats = [...it.desk.seats].sort((a, b) =>
        a.side === 'left' ? -1 : 1
      );
      html += `<div class="desk-seats">`;
      seats.forEach((seat) => {
        const status = seat.occupied ? 'occupied' : 'free';
        const sideText = seat.side === 'left' ? 'левое' : 'правое';
        const title = `Ряд ${ri + 1}, парта ${it.desk.index}, ${sideText} место — ${seat.occupied ? 'занято' : 'свободно'}`;
        html += `<div class="seat ${status}" title="${title}"></div>`;
      });
      html += `</div>`;
      // Визуальный разделитель между партами (вертикальный отступ)
      if (deskIdx < row.length - 1) {
        html += `<div class="desk-gap"></div>`;
      }
    });
    html += `</div></div>`;
  });

  seatingChartEl.innerHTML = html;
}

function renderResult(data) {
  freeSeatsEl.textContent = data.free_seats;
  occupiedSeatsEl.textContent = data.occupied_seats;
  totalSeatsEl.textContent = data.total_seats;
  desksDetectedEl.textContent = data.desks_detected;
  noteEl.textContent = data.note || "";
  annotatedImage.src = data.annotated_image;

  renderSeatingChart(data.desks);

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
