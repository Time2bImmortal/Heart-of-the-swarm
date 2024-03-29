#include <hidboot.h>
#include <usbhub.h>

int threshold = 2;

class MouseRptParser : public MouseReportParser {
  protected:
    void OnMouseMove(MOUSEINFO *mi);
    void OnLeftButtonUp(MOUSEINFO *mi) {}
    void OnLeftButtonDown(MOUSEINFO *mi) {}
    void OnRightButtonUp(MOUSEINFO *mi) {}
    void OnRightButtonDown(MOUSEINFO *mi) {}
    void OnMiddleButtonUp(MOUSEINFO *mi) {}
    void OnMiddleButtonDown(MOUSEINFO *mi) {}
};

void MouseRptParser::OnMouseMove(MOUSEINFO *mi) {
  int magnitude = sqrt(pow(mi->dX, 2) + pow(mi->dY, 2));
  
  if (magnitude < threshold) {
    Serial.println("0,0");
  } else {
    Serial.print(abs(mi->dX), DEC);
    Serial.print(",");
    Serial.println(abs(mi->dY), DEC);
  }
  
  delay(100);
}

USB Usb;
USBHub Hub(&Usb);
HIDBoot<USB_HID_PROTOCOL_MOUSE> HidMouse(&Usb);

MouseRptParser Prs;
uint32_t next_time;

void setup() {
  Serial.begin(115200);

  if (Usb.Init() == -1) {
    delay(200);
  }

  HidMouse.SetReportParser(0, &Prs);
}

void loop() {
  Usb.Task();
}
