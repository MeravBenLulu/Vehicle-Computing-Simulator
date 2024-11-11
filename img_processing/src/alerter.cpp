#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "alerter.h"
#include "alert.h"
#include "manager.h"

using json = nlohmann::json;
using namespace std;

// create buffer from alert object
vector<uint8_t> Alerter::makeAlertBuffer(int type, float distance,
                                         float relativeVelocity)
{
    Alert alert(false, 1, type, distance, relativeVelocity);
    vector<uint8_t> serialized = alert.serialize();
    return serialized;
}

void Alerter::destroyAlertBuffer(char *buffer)
{
    delete[] buffer;
}

// create alerts buffer to send
vector<vector<uint8_t>> Alerter::sendAlerts(
    const vector<ObjectInformation> &output)
{
    vector<vector<uint8_t>> alerts;
    for (const ObjectInformation &objectInformation : output) {
        if (isSendAlert(objectInformation)) {
            vector<uint8_t> alertBuffer = makeAlertBuffer(
                objectInformation.type, objectInformation.distance,
                objectInformation.velocity.value());
            alerts.push_back(alertBuffer);
        }
    }
    return alerts;
}

// Check whether to send alert
bool Alerter::isSendAlert(const ObjectInformation &objectInformation)
{
    return objectInformation.distance < MIN_LEGAL_DISTANCE;
}