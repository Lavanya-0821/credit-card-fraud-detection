// ============================================================
// CREDIT CARD FRAUD DETECTION
// FRONTEND JAVASCRIPT
// ============================================================


const form =
    document.getElementById("fraudForm");

const sampleButton =
    document.getElementById("sampleButton");

const resultBox =
    document.getElementById("result");

const loading =
    document.getElementById("loading");


// ============================================================
// LOAD SAMPLE TRANSACTION
// ============================================================

sampleButton.addEventListener(
    "click",
    async function () {

        try {

            sampleButton.disabled = true;

            sampleButton.innerHTML =
                "⏳ Loading...";


            const response =
                await fetch("/sample");


            if (!response.ok) {

                throw new Error(
                    "Server error"
                );

            }


            const data =
                await response.json();


            // Fill all inputs

            for (
                const feature in data
            ) {

                const input =
                    document.querySelector(
                        `[name="${feature}"]`
                    );


                if (input) {

                    input.value =
                        data[feature];

                }

            }


            sampleButton.innerHTML =
                "✅ Sample Loaded";


            // Reset button after 2 seconds

            setTimeout(
                function () {

                    sampleButton.innerHTML =
                        "📥 Load Sample";

                },
                2000
            );


        }

        catch (error) {

            console.error(error);

            alert(
                "Unable to load sample transaction."
            );


            sampleButton.innerHTML =
                "📥 Load Sample";

        }

        finally {

            sampleButton.disabled =
                false;

        }

    }
);


// ============================================================
// PREDICT TRANSACTION
// ============================================================

form.addEventListener(
    "submit",
    async function (event) {

        event.preventDefault();


        // Hide previous result

        resultBox.style.display =
            "none";


        // Show loading

        loading.style.display =
            "flex";


        // Disable predict button

        const predictButton =
            form.querySelector(
                ".predict-button"
            );

        predictButton.disabled =
            true;


        predictButton.innerHTML =
            "⏳ Analyzing...";


        // Collect form data

        const formData =
            new FormData(form);


        const transaction = {};


        formData.forEach(
            function (value, key) {

                transaction[key] =
                    parseFloat(value);

            }
        );


        try {

            // Send data to Flask

            const response =
                await fetch(
                    "/predict",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(
                                transaction
                            )
                    }
                );


            const data =
                await response.json();


            // Hide loading

            loading.style.display =
                "none";


            // Check error

            if (!data.success) {

                resultBox.className =
                    "result fraud";

                resultBox.innerHTML = `

                    <h3>
                        ❌ Prediction Error
                    </h3>

                    <p>
                        ${data.error}
                    </p>

                `;

                resultBox.style.display =
                    "block";

                return;

            }


            // ====================================================
            // FRAUD RESULT
            // ====================================================

            if (
                data.result === "FRAUD"
            ) {

                resultBox.className =
                    "result fraud";


                resultBox.innerHTML = `

                    <h3>
                        🚨 FRAUD TRANSACTION
                    </h3>

                    <p>
                        ${data.message}
                    </p>

                    <p>
                        Fraud Probability:
                        <strong>
                            ${data.probability}%
                        </strong>
                    </p>

                `;

            }


            // ====================================================
            // NORMAL RESULT
            // ====================================================

            else {

                resultBox.className =
                    "result normal";


                resultBox.innerHTML = `

                    <h3>
                        ✅ NORMAL TRANSACTION
                    </h3>

                    <p>
                        ${data.message}
                    </p>

                    <p>
                        Fraud Probability:
                        <strong>
                            ${data.probability}%
                        </strong>
                    </p>

                `;

            }


            // Show result

            resultBox.style.display =
                "block";


            // Scroll to result

            resultBox.scrollIntoView({
                behavior: "smooth",
                block: "center"
            });


        }

        catch (error) {

            console.error(error);


            loading.style.display =
                "none";


            resultBox.className =
                "result fraud";


            resultBox.innerHTML = `

                <h3>
                    ❌ Server Error
                </h3>

                <p>
                    Please make sure Flask
                    server is running.
                </p>

            `;


            resultBox.style.display =
                "block";

        }


        finally {

            // Enable button

            predictButton.disabled =
                false;


            predictButton.innerHTML =
                "🔎 Predict Transaction";

        }

    }
);