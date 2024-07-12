package com.example.mastersproject

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import okhttp3.*
import org.json.JSONObject
import java.io.IOException

class MetricsActivity : AppCompatActivity() {

    private lateinit var metricsTextView: TextView
    private lateinit var backButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_metrics)

        metricsTextView = findViewById(R.id.metrics_text_view)
        backButton = findViewById(R.id.back_button)

        fetchMetrics()

        backButton.setOnClickListener {
            finish()
        }
    }

    private fun fetchMetrics() {
        val url = "https://glucose.dynv6.net/metrics"  // Update with your Flask app URL
        Log.d("MetricsActivity", "Fetching metrics from URL: $url")

        val request = Request.Builder().url(url).build()
        val client = OkHttpClient()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                e.printStackTrace()
                Log.e("MetricsActivity", "Error fetching metrics: $e")
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    val responseData = response.body?.string()
                    Log.d("MetricsActivity", "Response data: $responseData")

                    if (responseData != null) {
                        runOnUiThread {
                            try {
                                val json = JSONObject(responseData)

                                val lstmMetrics = json.getJSONObject("LSTM")
                                val rfMetrics = json.getJSONObject("Random Forest")
                                val xgbMetrics = json.getJSONObject("XGBoost")

                                val metrics = StringBuilder()

                                metrics.append("LSTM Metrics:\n")
                                metrics.append("MAE: ").append(lstmMetrics.getDouble("MAE")).append("\n")
                                metrics.append("MAPE: ").append(lstmMetrics.getDouble("MAPE")).append("\n")
                                metrics.append("MSE: ").append(lstmMetrics.getDouble("MSE")).append("\n")
                                metrics.append("RMSE: ").append(lstmMetrics.getDouble("RMSE")).append("\n\n")

                                metrics.append("Random Forest Metrics:\n")
                                metrics.append("MAE: ").append(rfMetrics.getDouble("MAE")).append("\n")
                                metrics.append("MAPE: ").append(rfMetrics.getDouble("MAPE")).append("\n")
                                metrics.append("MSE: ").append(rfMetrics.getDouble("MSE")).append("\n")
                                metrics.append("RMSE: ").append(rfMetrics.getDouble("RMSE")).append("\n\n")

                                metrics.append("XGBoost Metrics:\n")
                                metrics.append("MAE: ").append(xgbMetrics.getDouble("MAE")).append("\n")
                                metrics.append("MAPE: ").append(xgbMetrics.getDouble("MAPE")).append("\n")
                                metrics.append("MSE: ").append(xgbMetrics.getDouble("MSE")).append("\n")
                                metrics.append("RMSE: ").append(xgbMetrics.getDouble("RMSE")).append("\n")

                                metricsTextView.text = metrics.toString()
                            } catch (e: Exception) {
                                e.printStackTrace()
                                Log.e("MetricsActivity", "Error parsing metrics: $e")
                            }
                        }
                    } else {
                        Log.e("MetricsActivity", "Empty response data")
                    }
                } else {
                    Log.e("MetricsActivity", "Unsuccessful response: ${response.code}")
                }
            }
        })
    }
}