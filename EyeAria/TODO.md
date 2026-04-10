# Plans

### Phase 1: Hardening the Architecture (Modularity & Extendability)

* **Define a Strict Data Schema:** Currently, nodes pass data via `json.loads(data_json)` and `json.dumps()`.
  - **TODO:** Implement a schema validation layer (like Pydantic) to ensure every node outputs and expects a standardized payload format (e.g., standardizing the `detections` array and `track_id`).
* **Headless Runtime Mode:** Right now, the pipeline is tightly coupled to the NiceGUI frontend.
  - **TODO:** Create a script (e.g., `headless_runner.py`) that can load a saved `full_pipeline_config.json` and execute the `start()` loops without spinning up the web server.
* **Plugin Error Isolation:**
  - **TODO:** Wrap the `_input` and `_start` execution of user-created plugins in robust `try/except` blocks so one poorly written plugin doesn't crash the entire `PipelineApp`.

---

### Phase 2: Testing Framework

* **Unit Test the Graph Logic:**
  - **TODO:** Write `pytest` scripts that programmatically instantiate nodes, connect them using `add_subscriber`, inject a mock JSON payload into the source, and assert the output of the sink.
* **Pipeline Component Tests:**
  - **TODO:** Isolate `HailoPipeline.py` and write tests using file sources (`filesrc`) to guarantee your GStreamer strings compile and push buffers correctly without needing a physical camera attached.
* **Hardware State Tests:**
 - **TODO:** Expand on your existing `[MOCK]` servo logic. Write tests that simulate a detection payload and assert that the `enter_sequence` and `exit_sequence` trigger the correct mock pins at the correct time intervals.

---

### Phase 3: Deployment & Distribution

* **Dockerization:**
  - **TODO:** Write a `Dockerfile`. This is crucial because GStreamer, Hailo dependencies (`.hef`, `.so` files), and OpenCV can be an absolute nightmare to install natively on every new device.
* **Systemd Service Integration:**
  - **TODO:** Create a `.service` file template so users can easily set your headless runner to start automatically on boot.
* **Configuration Management:**
  - **TODO:** Extract hardcoded defaults into `.env` files or a global settings menu.

---

### Phase 4: Extra Plugins (Ideas for Expansion)

| Plugin Idea | Description |
| --- | --- |
| **Webhooks / REST Sink** | A node to send `POST` requests to external APIs (like Slack, Discord, or HomeAssistant) when specific conditions are met. |
| **Database Sink** | A node that connects to SQLite or PostgreSQL to log historical detection data over time. |
| **Math / Logic Gate Node** | A node that can aggregate data from *multiple* sources (e.g., waiting for both Camera A and Camera B to trigger before sending a signal). |
| **Crop & Save Node** | A node that extracts the raw frame, crops it using the `bbox` coordinates, and saves the image of the detected object locally. |

---

# Priority Order

### Priority 1: Core Engine Stability & Unit Tests (The Foundation)

Before adding more features or deploying, we need to ensure the existing graph logic won't collapse if a bad payload is sent.

* **Decouple the Runner:** Build a headless runner script that loads your saved JSON pipeline and calls `start()` on the root node, completely bypassing the NiceGUI interface ``. This ensures your pipeline runs even if the web server crashes.
* **Unit Test Graph Execution:** Write `pytest` scripts that programmatically instantiate a few nodes, connect them using `add_subscriber()`, and pass mock JSON strings through `notify()` ``. Verify that data propagates correctly down the chain.
* **Standardize Data Schema:** Enforce a strict schema (like Pydantic) for your JSON payloads so every node knows exactly what keys to expect in the `detections` array ``.
* **Plugin Isolation:** Wrap the `_input()` and `_start()` methods in robust `try/except` blocks ``. If a custom plugin fails, it should log an error, not take down the entire `PipelineApp`.

### Priority 2: Hardware Integration Tests (The Physical Layer)

Since you have the hardware, let's put it to work. Integration tests ensure your software talks to the Hailo chip and GPIO pins correctly.

* **Pipeline File-Source Testing:** Write integration tests that instantiate `HailoPipeline` using a `FileSource` (local video) instead of a live camera ``. Assert that the `AppSink` successfully parses the Hailo metadata and fires the callback with populated bounding boxes.
* **GPIO Hardware Validation:** Create a test suite specifically for `GPIONode` ``. Feed it a stream of mock detections that trigger an `enter_sequence`, and use a multimeter or LED on your Pi to verify the `gpiozero` pins are actually going HIGH/LOW as expected.
* **MQTT End-to-End Test:** Spin up a local Mosquitto broker on the Pi. Have your `MQTTSource` node `listen to a topic, and your `MqttSink` node` publish to it. Verify the round-trip latency and payload integrity.

### Priority 3: Deployment & Autonomy (The Edge Ready Phase)

Now that the code is tested and runs headlessly, we need to make it survive reboots and run smoothly on the Pi.

* **Systemd Service:** Create a `.service` file that automatically launches your headless runner script on boot. This is critical for edge devices that might lose power.
* **Dockerization (Optional but Recommended):** Write a `Dockerfile`. Packaging GStreamer, the HailoRT dependencies (`.hef`, `.so` files ``), and OpenCV into a container will save you hours of dependency hell if you ever need to deploy this to a *second* Raspberry Pi.
* **Environment Variables:** Move hardcoded credentials (like `broker.hivemq.com` or MQTT passwords ``) into a `.env` file or a global settings menu in the UI.

### Priority 4: Expansion (The Fun Stuff)

Once the system is bulletproof and deployed, you can safely build out your plugin ecosystem.

* **Multi-Cam Logic Gates:** A node that takes inputs from two different Hailo cameras and requires an `AND` condition before firing the GPIO pin.
* **Database Logging:** A sink node that writes tracking history to a local SQLite database for later analysis.
* **Notification Webhooks:** A sink node that sends a POST request to Discord or Slack when a specific object (like a person) is detected.
